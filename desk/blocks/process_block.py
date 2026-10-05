# =====================================================================
# FILE: blocks/process_block.py
# =====================================================================
from desk.core.base_block import BaseBlock
from desk.core.entity import Entity, EventLogger
from typing import Dict, Callable, Optional
import simpy


class ProcessBlock(BaseBlock):
    """
    PROCESS block - performs delay operation with optional resource seizure.
    
    Can operate in two modes:
    1. With resource: seize resource, delay, release resource (traditional queue)
    2. Without resource: pure delay operation (no queueing)

    BLOCKING BEHAVIOR (blocking-after-service):
        By default (blocking=True), a resource-holding ProcessBlock will NOT
        release its resource the instant its own service finishes. Instead
        it first tries to get the entity admitted into whatever comes next
        (following the connected block, and transparently through any
        DecideBlock routing). Only once that admission succeeds does it
        release its own resource.

        This means: if the next stage (or the stage after a routing
        decision) is a ProcessBlock whose resource is fully occupied, THIS
        block stays occupied too -- it is "blocked" holding its resource
        with a finished entity that has nowhere to go, exactly like a
        patient who can't be moved out of a PS bed because the ward is
        full. That, in turn, makes upstream resources back up as well,
        since nothing here frees up for new arrivals.

        Set blocking=False to opt back into the old fire-and-forget
        behavior (release immediately, forward asynchronously into an
        effectively unlimited downstream queue).

    OPTIONAL FINITE BUFFER (buffer_capacity):
        By default the only queue in front of a resource is SimPy's own
        (unbounded) resource.queue -- i.e. entities can queue here without
        limit before upstream ever blocks. Setting buffer_capacity=N adds
        a genuine finite "waiting room" (e.g. a corridor with N stretcher
        slots) in front of the resource: once N entities are already
        waiting AND the resource itself is full, any further arrival
        (and, transitively, whatever is upstream trying to hand off here)
        blocks until a slot frees up.
    
    Args:
        name: Block name
        env: SimPy environment
        delay_time: Function returning delay duration
        resource: Optional resource to seize (None = pure delay)
        resource_units: Number of resource units to seize (default 1)
        event_logger: Optional event logger
        blocking: If True (default), hold this block's resource until the
            entity is admitted downstream (see above). If False, release
            immediately and forward asynchronously (legacy behavior).
        buffer_capacity: Optional finite waiting-room size in front of the
            resource (see above). None = unbounded (SimPy default queue).
    """
    
    def __init__(self, name: str, env: simpy.Environment,
                 delay_time: Callable[[], float],
                 resource: Optional[simpy.Resource] = None,
                 resource_units: int = 1,
                 event_logger: EventLogger = None,
                 blocking: bool = True,
                 buffer_capacity: Optional[int] = None):
        super().__init__(name, env, event_logger)
        self.resource = resource
        self.delay_time = delay_time
        self.resource_units = resource_units
        self.entities_processed = 0
        self.total_delay_time = 0.0
        self.total_queue_time = 0.0
        self.resource_data = []  # (time, in_service, queue_length)
        self.max_queue_length = 0
        self.max_in_service = 0        
        self.resource_name = None # Store resource name for logging

        # --- Blocking / finite-buffer configuration ---
        self.blocking = blocking
        self.buffer_capacity = buffer_capacity
        self.buffer: Optional[simpy.Resource] = (
            simpy.Resource(env, capacity=buffer_capacity)
            if (resource is not None and buffer_capacity is not None)
            else None
        )
        self.max_buffer_length = 0

        # Optional instrumentation hooks (no-ops unless set). These fire at
        # the exact moment an entity has finished service here but is
        # stuck holding this block's resource because the next block
        # hasn't admitted it yet, and again when it's finally admitted.
        # External tooling (e.g. interface.py's live VisualizationInstrument,
        # or a custom logger) can set these without needing to patch any
        # private method, since the "blocked, still holding my resource"
        # state only exists inside the background service coroutine and
        # isn't otherwise observable from outside.
        self.on_block_wait: Optional[Callable[[Entity], None]] = None
        self.on_block_release: Optional[Callable[[Entity], None]] = None
    
    def set_resource_name(self, name: str):
        """Set the resource name for event logging."""
        self.resource_name = name

    def process_entity(self, entity: Entity):
        """
        Top-level entry point for an entity arriving at this block.

        Delegates to request_admission(): for a resource-less block this
        completes (and runs the delay) right away; for a resource-holding
        block this waits until the resource is actually secured, then lets
        the rest of the stay run in the background.
        """
        yield from self.request_admission(entity)

    def request_admission(self, entity: Entity):
        """
        Secure a spot for `entity` at this block.

        - No resource configured: admission is instantaneous (unlimited
          capacity); the delay + forwarding runs as an independent process.
        - Resource configured: waits (optionally behind a finite buffer)
          until the resource is actually acquired, then hands the rest of
          the stay (service + eventual blocking hand-off to whatever's
          next) to an independent background process and returns. This is
          what lets an upstream caller release ITS resource the moment
          this call returns, without needing to sit through this block's
          entire service time.
        """
        entity.route_history.append(self.name)

        if self.resource is None:
            # No capacity constraint here -- always instant admission.
            self.env.process(self._process_without_resource(entity))
            return

        self._monitor_resource()

        # Optional finite waiting room in front of the resource. Only once
        # BOTH the buffer and the resource are full does a caller actually
        # block on this yield.
        buffer_req = None
        if self.buffer is not None:
            buffer_req = self.buffer.request()
            self._trace('queue', entity, self.resource_name,
                        f"waiting for buffer slot, buffer_length={len(self.buffer.queue)}")
            yield buffer_req
            self.max_buffer_length = max(self.max_buffer_length, self.buffer.count)

        queue_start = self.env.now
        request_priority = (self.activity_priority
                             if self.activity_priority is not None
                             else entity.priority)

        requests = self._make_requests(request_priority)

        queue_length = len(self.resource.queue)
        self._trace('queue', entity, self.resource_name,
                    f"waiting, queue_length={queue_length}")

        # ACQUISITION - this is where an upstream, still-resource-holding
        # caller actually blocks if this resource is full.
        yield simpy.AllOf(self.env, requests)

        # Secured: release the buffer slot (moved from waiting room into
        # service) and let the rest of this entity's stay run in the
        # background so this call can return now.
        if buffer_req is not None:
            self.buffer.release(buffer_req)

        self._monitor_resource()
        queue_time = self.env.now - queue_start

        self.env.process(self._serve_with_resource(entity, requests, queue_time))

    def _make_requests(self, request_priority: int):
        """Build the list of resource requests according to resource_units."""
        requests = []
        for _ in range(self.resource_units):
            if isinstance(self.resource, simpy.PreemptiveResource):
                # Use preempt=False during request; preemption still
                # occurs during the service timeout.
                req = self.resource.request(priority=request_priority, preempt=False)
            elif isinstance(self.resource, simpy.PriorityResource):
                req = self.resource.request(priority=request_priority)
            else:
                req = self.resource.request()
            requests.append(req)
        return requests

    def _process_without_resource(self, entity: Entity):
        """Process entity with pure delay (no resource seizure)."""
        # Log activity start
        self.log_start(entity, resource_name=None)
        
        # Calculate delay
        if hasattr(self.env, 'model') and hasattr(self.env.model, 'safe_delay_time'):
            delay = self.env.model.safe_delay_time(self.delay_time)
        else:
            delay = max(0.0, self.delay_time())
        
        # Perform delay
        yield self.env.timeout(delay)
        
        # Update statistics
        self.entities_processed += 1
        self.total_delay_time += delay
        entity.add_attribute(f"{self.name}_service_time", delay)
        entity.add_attribute(f"{self.name}_queue_time", 0.0)  # No queueing
        
        # Apply configured attributes
        self._apply_attributes(entity)
        
        # Log activity complete
        self.log_complete(entity, resource_name=None)
        
        # Continue to next block. No resource is held here, so there is
        # nothing to "block" -- just chain straight through. Note this
        # goes through request_admission (not send_to_next/process_entity)
        # so that EVERY internal hop -- resource or no resource -- uses the
        # same single entry point. That matters for anything instrumenting
        # the block graph (see interface.py's VisualizationInstrument): it
        # only has to wrap one method to see every hop consistently.
        if self.next_block is not None:
            yield from self.next_block.request_admission(entity)

    def _serve_with_resource(self, entity: Entity, requests, queue_time: float):
        """
        Runs after the resource has already been acquired: does the actual
        service, then (if blocking) waits for downstream admission BEFORE
        releasing the resource, then releases it.
        """
        acquired = requests

        while True:  # Retry loop for preemption during service
            try:
                self.total_queue_time += queue_time
                entity.add_attribute(f"{self.name}_queue_time", queue_time)

                # SERVICE - can be preempted here
                if hasattr(self.env, 'model') and hasattr(self.env.model, 'safe_delay_time'):
                    delay = self.env.model.safe_delay_time(self.delay_time)
                else:
                    delay = max(0.0, self.delay_time())

                utilization = self.resource.count / self.resource.capacity
                self._trace('service_start', entity, self.resource_name,
                           f"service_time={delay:.2f}, queue_time={queue_time:.2f}")

                self.log_start(entity, self.resource_name)

                yield self.env.timeout(delay)

                # SUCCESS - completed without interruption
                self.entities_processed += 1
                self.total_delay_time += delay
                entity.add_attribute(f"{self.name}_service_time", delay)

                # Capture assigned attributes
                assigned_attrs = self._apply_attributes(entity)
                modified_attrs = self._modify_attributes(entity)

                # Include attributes in trace
                utilization = self.resource.count / self.resource.capacity
                details = f"use={utilization:.0%}"

                # Collect all attribute changes
                attr_changes = []

                if assigned_attrs:
                    for name, value in assigned_attrs:
                        if isinstance(value, float):
                            attr_changes.append(f"{name}={value:.2f}")
                        else:
                            attr_changes.append(f"{name}={value}")

                if modified_attrs:
                    for name, old_val, new_val in modified_attrs:
                        if isinstance(new_val, float):
                            attr_changes.append(f"{name}: {old_val:.2f}\u2192{new_val:.2f}")
                        else:
                            attr_changes.append(f"{name}: {old_val}\u2192{new_val}")

                if attr_changes:
                    details += f", Attrib: {', '.join(attr_changes)}"

                self._trace('service_end', entity, self.resource_name, details)

                self.log_complete(entity, self.resource_name)

                break  # Exit retry loop - service is done

            except simpy.Interrupt:
                # Trace preemption
                self._trace('interrupt', entity, self.resource_name,
                           f"preempted by higher priority")

                if self.event_logger:
                    self.event_logger.log_event(
                        case_id=entity.id,
                        activity=self.name,
                        timestamp=self.env.now,
                        lifecycle='interrupt',
                        resource=self.resource_name,
                        priority=entity.priority,
                        activity_priority=self.activity_priority
                    )

                # Release what we had and re-acquire from scratch, then retry
                for req in acquired:
                    try:
                        self.resource.release(req)
                    except Exception:
                        pass
                self._monitor_resource()

                queue_start = self.env.now
                request_priority = (self.activity_priority
                                     if self.activity_priority is not None
                                     else entity.priority)
                requests = self._make_requests(request_priority)
                yield simpy.AllOf(self.env, requests)
                acquired = requests
                self._monitor_resource()
                queue_time = self.env.now - queue_start
                continue

        # ---- BLOCKING HAND-OFF ----
        # Do NOT release this resource until the entity has secured its
        # next spot downstream (or reached a block with no capacity
        # constraint, e.g. Dispose or a pure-delay block). This is what
        # makes a saturated downstream stage propagate backpressure into
        # this one.
        if self.next_block is not None:
            if self.blocking:
                # From this point until the yield below returns, `entity`
                # has finished service here but has nowhere to go yet --
                # this block's resource is still held. Trace it explicitly
                # so console tracing / replay and any external
                # instrumentation (see interface.py) can distinguish this
                # "blocked, holding resource" state from ordinary queueing
                # or active service.
                still_blocked = len(getattr(self.next_block, 'resource', None).queue) if getattr(self.next_block, 'resource', None) is not None else None
                self._trace('blocked', entity, self.resource_name,
                           f"finished, waiting on {self.next_block.name} "
                           f"(downstream_queue={still_blocked})" if still_blocked is not None
                           else f"finished, waiting to enter {self.next_block.name}")
                if self.on_block_wait:
                    self.on_block_wait(entity)

                yield from self.next_block.request_admission(entity)

                self._trace('unblocked', entity, self.resource_name,
                           f"admitted into {self.next_block.name}, releasing")
                if self.on_block_release:
                    self.on_block_release(entity)
            else:
                self.env.process(self.next_block.process_entity(entity))

        # Now safe to free the resource this entity was holding here.
        for req in acquired:
            try:
                self.resource.release(req)
            except Exception:
                pass
        self._monitor_resource()

    def _monitor_resource(self):
        """Monitor resource state for statistics (only if resource exists)."""
        if self.resource is None:
            return  # Skip monitoring if no resource
        
        current_queue_length = len(self.resource.queue)
        current_in_service = self.resource.count
        
        self.max_queue_length = max(self.max_queue_length, current_queue_length)
        self.max_in_service = max(self.max_in_service, current_in_service)
        
        # Always collect data for warm-up analysis
        data_point = (self.env.now, current_in_service, current_queue_length)
        self.resource_data.append(data_point)


class MultiProcessBlock(BaseBlock):
    """
    PROCESS block that seizes MULTIPLE resources simultaneously with
    activity priority.

    Participates in the same blocking-after-service / finite-buffer /
    instrumentation-hook mechanism as ProcessBlock (see its docstring for
    the full explanation). The only difference is that "this block's
    resource" means "all resources in resource_requirements, acquired
    together" -- admission is only granted once every required resource
    has a free unit, and the block doesn't release ANY of them until the
    entity is admitted downstream (or reaches a block with no capacity
    constraint).

    Args:
        name: Block name
        env: SimPy environment
        resource_requirements: Dict mapping resources to units needed,
            e.g. {nurses: 1, doctors: 1, pharmacy_staff: 1}
        delay_time: Function returning service time
        event_logger: Optional event logger
        blocking: If True (default), hold all of this block's resources
            until the entity is admitted downstream. If False, release
            immediately and forward asynchronously (legacy behavior).
        buffer_capacity: Optional finite waiting-room size gating entry
            into the multi-resource acquisition (None = unbounded).
    """
    
    def __init__(self, name: str, env: simpy.Environment,
                 resource_requirements: Dict[simpy.Resource, int],
                 delay_time: Callable[[], float],
                 event_logger: EventLogger = None,
                 blocking: bool = True,
                 buffer_capacity: Optional[int] = None):
        """
        Args:
            resource_requirements: Dict mapping resources to units needed
                                 e.g., {nurses: 1, doctors: 1, pharmacy_staff: 1}
            delay_time: Function returning service time
        """
        super().__init__(name, env, event_logger)
        self.resource_requirements = resource_requirements
        self.delay_time = delay_time
        self.entities_processed = 0
        self.resource_names = {}
        self.total_delay_time = 0.0
        self.total_queue_time = 0.0
        self.resource_data = {}  # Dict of resource -> [(time, in_service, queue_length)]
        self.max_metrics = {}    # Dict of resource -> {max_queue, max_service}
        
        # Initialize monitoring for each resource
        for resource in resource_requirements.keys():
            self.resource_data[resource] = []
            self.max_metrics[resource] = {'max_queue_length': 0, 'max_in_service': 0}

        # --- Blocking / finite-buffer configuration (mirrors ProcessBlock) ---
        self.blocking = blocking
        self.buffer_capacity = buffer_capacity
        self.buffer: Optional[simpy.Resource] = (
            simpy.Resource(env, capacity=buffer_capacity)
            if buffer_capacity is not None else None
        )
        self.max_buffer_length = 0

        # Optional instrumentation hooks -- see ProcessBlock for details.
        self.on_block_wait: Optional[Callable[[Entity], None]] = None
        self.on_block_release: Optional[Callable[[Entity], None]] = None
        
    def set_resource_names(self, resource_names: Dict[simpy.Resource, str]):
        """Set resource names for logging."""
        self.resource_names = resource_names

    def process_entity(self, entity: Entity):
        """Top-level entry point: delegates to request_admission()."""
        yield from self.request_admission(entity)

    def request_admission(self, entity: Entity):
        """
        Secure ALL required resources for `entity` at this block.

        Waits (optionally behind a finite buffer) until every resource in
        resource_requirements has a free unit, acquiring them all
        simultaneously (simpy.AllOf), then hands the rest of the stay
        (service + eventual blocking hand-off) to an independent
        background process and returns -- exactly like
        ProcessBlock.request_admission, just for a set of resources
        instead of one.
        """
        entity.route_history.append(self.name)

        queue_start = self.env.now
        self._monitor_all_resources()

        # Optional finite waiting room gating entry into the multi-resource
        # acquisition.
        buffer_req = None
        if self.buffer is not None:
            buffer_req = self.buffer.request()
            self._trace('queue', entity, "buffer",
                        f"waiting for buffer slot, buffer_length={len(self.buffer.queue)}")
            yield buffer_req
            self.max_buffer_length = max(self.max_buffer_length, self.buffer.count)

        request_priority = (self.activity_priority
                             if self.activity_priority is not None
                             else entity.priority)
        requests = self._make_requests(request_priority)

        resources_str = ", ".join([self.resource_names.get(r, "Unknown")
                                  for r, _ in requests])
        total_queue_length = sum(len(r.queue) for r, _ in requests)
        self._trace('queue', entity, resources_str,
                   f"waiting for all resources, total_queue={total_queue_length}")

        # ACQUISITION - this is where an upstream, still-resource-holding
        # caller actually blocks if any required resource is full.
        yield simpy.AllOf(self.env, [req for _, req in requests])

        if buffer_req is not None:
            self.buffer.release(buffer_req)

        acquired_resources = requests
        self._monitor_all_resources()

        self.env.process(self._serve_with_resources(entity, acquired_resources, queue_start))

    def _make_requests(self, request_priority: int):
        """Build the list of (resource, request) pairs for all required resources."""
        requests = []
        for resource, units in self.resource_requirements.items():
            for _ in range(units):
                if isinstance(resource, simpy.PreemptiveResource):
                    req = resource.request(priority=request_priority, preempt=True)
                elif isinstance(resource, simpy.PriorityResource):
                    req = resource.request(priority=request_priority)
                else:
                    req = resource.request()
                requests.append((resource, req))
        return requests

    def _serve_with_resources(self, entity: Entity, acquired_resources, queue_start: float):
        """
        Runs after all required resources have already been acquired: does
        the actual service, then (if blocking) waits for downstream
        admission BEFORE releasing any resource, then releases them all.
        """
        resources_str = ", ".join([self.resource_names.get(r, "Unknown")
                                  for r, _ in acquired_resources])

        while True:  # Retry loop for preemption during service
            try:
                # Record queue time and monitor state after seizing all
                queue_time = self.env.now - queue_start
                self.total_queue_time += queue_time
                entity.add_attribute(f"{self.name}_queue_time", queue_time)

                resources_str = ", ".join([self.resource_names.get(r, "Unknown")
                                          for r, _ in acquired_resources])
                self.log_start(entity, resources_str)

                if hasattr(self.env, 'model') and hasattr(self.env.model, 'safe_delay_time'):
                    delay = self.env.model.safe_delay_time(self.delay_time)
                else:
                    delay = max(0.0, self.delay_time())

                avg_utilization = sum(r.count / r.capacity for r, _ in acquired_resources) / len(acquired_resources)
                self._trace('service_start', entity, resources_str,
                           f"service_time={delay:.2f}, queue_time={queue_time:.2f}")
                self.log_start(entity, resources_str)

                yield self.env.timeout(delay)

                # SUCCESS
                self.entities_processed += 1
                self.total_delay_time += delay
                entity.add_attribute(f"{self.name}_service_time", delay)

                assigned_attrs = self._apply_attributes(entity)
                modified_attrs = self._modify_attributes(entity)

                avg_utilization = sum(r.count / r.capacity for r, _ in acquired_resources) / len(acquired_resources)
                details = f"use={avg_utilization:.0%}"

                attr_changes = []
                if assigned_attrs:
                    for name, value in assigned_attrs:
                        if isinstance(value, float):
                            attr_changes.append(f"{name}={value:.2f}")
                        else:
                            attr_changes.append(f"{name}={value}")
                if modified_attrs:
                    for name, old_val, new_val in modified_attrs:
                        if isinstance(new_val, float):
                            attr_changes.append(f"{name}: {old_val:.2f}\u2192{new_val:.2f}")
                        else:
                            attr_changes.append(f"{name}: {old_val}\u2192{new_val}")
                if attr_changes:
                    details += f", Attrib: {', '.join(attr_changes)}"

                self._trace('service_end', entity, resources_str, details)
                self.log_complete(entity, resources_str)

                break  # Success, exit retry loop

            except simpy.Interrupt:
                self._trace('interrupt', entity, resources_str,
                           f"preempted by higher priority")

                if self.event_logger:
                    self.event_logger.log_event(
                        case_id=entity.id,
                        activity=self.name,
                        timestamp=self.env.now,
                        lifecycle='interrupt',
                        resource=resources_str,
                        priority=entity.priority,
                        activity_priority=self.activity_priority
                    )

                # Release everything we had and re-acquire all resources
                # from scratch, then retry service.
                for resource, req in acquired_resources:
                    try:
                        resource.release(req)
                    except Exception:
                        pass
                self._monitor_all_resources()

                request_priority = (self.activity_priority
                                     if self.activity_priority is not None
                                     else entity.priority)
                new_requests = self._make_requests(request_priority)
                yield simpy.AllOf(self.env, [req for _, req in new_requests])
                acquired_resources = new_requests
                self._monitor_all_resources()
                continue

        # ---- BLOCKING HAND-OFF ----
        # Do NOT release any of these resources until the entity has
        # secured its next spot downstream. This is what makes a
        # saturated downstream stage propagate backpressure into every
        # resource this block holds -- e.g. if nurses+doctors are both
        # tied up here because the next ward is full, neither frees up for
        # a new arrival until that ward has room.
        if self.next_block is not None:
            if self.blocking:
                self._trace('blocked', entity, resources_str,
                           f"finished, waiting on {self.next_block.name}")
                if self.on_block_wait:
                    self.on_block_wait(entity)

                yield from self.next_block.request_admission(entity)

                self._trace('unblocked', entity, resources_str,
                           f"admitted into {self.next_block.name}, releasing")
                if self.on_block_release:
                    self.on_block_release(entity)
            else:
                self.env.process(self.next_block.process_entity(entity))

        # Now safe to free every resource this entity was holding here.
        for resource, req in acquired_resources:
            try:
                resource.release(req)
            except Exception:
                pass
        self._monitor_all_resources()

    def _monitor_all_resources(self):
        """Monitor state of all resources."""
        for resource in self.resource_requirements.keys():
            current_queue_length = len(resource.queue)
            current_in_service = resource.count
            
            # Update max metrics
            self.max_metrics[resource]['max_queue_length'] = max(
                self.max_metrics[resource]['max_queue_length'], 
                current_queue_length
            )
            self.max_metrics[resource]['max_in_service'] = max(
                self.max_metrics[resource]['max_in_service'], 
                current_in_service
            )
            
            # Store data point
            data_point = (self.env.now, current_in_service, current_queue_length)
            self.resource_data[resource].append(data_point)

