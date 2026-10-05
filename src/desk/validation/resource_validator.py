# =====================================================================
# FILE: validation/resource_validator.py
# =====================================================================
# Always validate (it's automatic by default)
# Use print_resource_summary() during development
# resource_units must be ≤ capacity (critical rule)
# Validation prevents deadlocks before they happen
# Clear error messages tell you exactly what to fix

# Validation Checks
# Check                   Level       Description
# Units > Capacity        ❌ ERROR    DEADLOCK - stops simulation
# Units == Capacity       ⚠️ WARNING  May create bottleneck
# Units > 50%             ℹ️ INFO      High resource usage
# Unregistered Resource   ❌ ERROR    Resource not found
# Wrong Type              ⚠️ WARNING  Priority mismatch

"""
Resource configuration validation for simulation models.

Validates that:
- Resource units requested don't exceed capacity
- Resources exist before being used
- No duplicate resource names
- Valid resource types
- Consistent resource usage across blocks
"""

from typing import Optional
import simpy


class ResourceValidationError(Exception):
    """Raised when resource configuration is invalid."""
    """
    Resource configuration validation for simulation models.
    Validates that:
    - Resource units requested don't exceed capacity
    - Resources exist before being used
    - No duplicate resource names
    - Valid resource types
    - Consistent resource usage across blocks
    """
    pass


class ResourceValidator:
    """
    Validates resource configurations in simulation models.
    
    Performs comprehensive checks to catch configuration errors before
    simulation runtime, providing clear error messages for fixes.
    
    Supports ProcessBlocks with and without resources (pure delay operations).
    """
    
    def __init__(self, model):
        """
        Initialize resource validator.
        
        Args:
            model: SimulationModel instance to validate
        """
        self.model = model
        self.errors = []
        self.warnings = []
    
    def validate_all(self, raise_on_error: bool = True) -> bool:
        """
        Run all validation checks.
        
        Args:
            raise_on_error: If True, raise exception on errors; if False, return status
            
        Returns:
            True if all validations pass, False otherwise
            
        Raises:
            ResourceValidationError: If validation fails and raise_on_error=True
        """
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._validate_resource_definitions()
        self._validate_resource_units()
        self._validate_resource_references()
        self._validate_resource_types()
        self._validate_multi_resource_blocks()
        self._check_blocking_deadlock_cycles()
        
        # Print results
        self._print_validation_results()
        
        # Handle errors
        if self.errors:
            if raise_on_error:
                error_msg = self._format_error_message()
                raise ResourceValidationError(error_msg)
            return False
        
        return True
    
    def _validate_resource_definitions(self):
        """Check for duplicate resource names and invalid capacities."""
        seen_names = set()
        
        for name, resource in self.model.resources.items():
            # Check for duplicates (shouldn't happen with dict, but check anyway)
            if name in seen_names:
                self.errors.append(
                    f"DUPLICATE RESOURCE: '{name}' is defined multiple times"
                )
            seen_names.add(name)
            
            # Check capacity
            if resource.capacity <= 0:
                self.errors.append(
                    f"INVALID CAPACITY: Resource '{name}' has capacity "
                    f"{resource.capacity} (must be > 0)"
                )
            
            # Warning for very high capacity
            if resource.capacity > 1000:
                self.warnings.append(
                    f"HIGH CAPACITY: Resource '{name}' has unusually high capacity "
                    f"({resource.capacity}). Is this intentional?"
                )
    
    def _validate_resource_units(self):
        """Validate that resource units don't exceed capacity."""
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        for block_name, block in self.model.blocks.items():
            if isinstance(block, ProcessBlock):
                self._validate_single_resource_block(block_name, block)
            elif isinstance(block, MultiProcessBlock):
                self._validate_multi_resource_block_units(block_name, block)
    
    def _validate_single_resource_block(self, block_name: str, block):
        """Validate ProcessBlock resource configuration."""
        resource = block.resource
        
        # Skip validation if block has no resource (pure delay mode)
        if resource is None:
            return  # No resource = no validation needed
        
        units_requested = getattr(block, 'resource_units', 1)
        
        # Find resource name
        resource_name = self._find_resource_name(resource)
        
        if resource_name:
            capacity = resource.capacity
            
            # CRITICAL ERROR: Units exceed capacity
            if units_requested > capacity:
                self.errors.append(
                    f"RESOURCE OVER ALLOCATION: Block '{block_name}' requests "
                    f"{units_requested} units of '{resource_name}', but capacity is "
                    f"only {capacity}. This will cause DEADLOCK!"
                )
            
            # WARNING: Using full capacity (might cause bottleneck)
            elif units_requested == capacity:
                self.warnings.append(
                    f"FULL RESOURCE USE: Block '{block_name}' uses ALL {capacity} "
                    f"units of '{resource_name}'. This may create a bottleneck."
                )
            
            # WARNING: Units > 50% of capacity
            elif units_requested > capacity * 0.5:
                utilization_pct = (units_requested / capacity) * 100
                self.warnings.append(
                    f"HIGH RESOURCE USE: Block '{block_name}' uses {units_requested} "
                    f"of {capacity} units ({utilization_pct:.0f}%) of '{resource_name}'. "
                    f"Consider if this is appropriate."
                )
        else:
            self.errors.append(
                f"UNKNOWN RESOURCE: Block '{block_name}' uses a resource that "
                f"is not registered in the model"
            )
    
    def _validate_multi_resource_block_units(self, block_name: str, block):
        """Validate MultiProcessBlock resource requirements."""
        for resource, units_requested in block.resource_requirements.items():
            resource_name = self._find_resource_name(resource)
            
            if resource_name:
                capacity = resource.capacity
                
                # CRITICAL ERROR: Units exceed capacity
                if units_requested > capacity:
                    self.errors.append(
                        f"RESOURCE OVERALLOCATION: Block '{block_name}' requests "
                        f"{units_requested} units of '{resource_name}', but capacity is "
                        f"only {capacity}. This will cause DEADLOCK!"
                    )
                
                # WARNING: Using full capacity
                elif units_requested == capacity:
                    self.warnings.append(
                        f"FULL RESOURCE USE: Block '{block_name}' uses ALL {capacity} "
                        f"units of '{resource_name}'. Combined with other resources, "
                        f"this may create significant bottleneck."
                    )
            else:
                self.errors.append(
                    f"UNKNOWN RESOURCE: Block '{block_name}' uses a resource that "
                    f"is not registered in the model"
                )
    
    def _validate_resource_references(self):
        """Check that all referenced resources exist."""
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        registered_resources = set(self.model.resources.values())
        
        for block_name, block in self.model.blocks.items():
            if isinstance(block, ProcessBlock):
                # Skip if no resource (pure delay mode)
                if block.resource is None:
                    continue
                
                if block.resource not in registered_resources:
                    self.errors.append(
                        f"UNREGISTERED RESOURCE: Block '{block_name}' uses a resource "
                        f"that was not added via model.add_resource()"
                    )
            
            elif isinstance(block, MultiProcessBlock):
                for resource in block.resource_requirements.keys():
                    if resource not in registered_resources:
                        self.errors.append(
                            f"UNREGISTERED RESOURCE: Block '{block_name}' uses a resource "
                            f"that was not added via model.add_resource()"
                        )
    
    def _validate_resource_types(self):
        """Validate resource types (Regular vs Priority vs Preemptive)."""
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        for block_name, block in self.model.blocks.items():
            if isinstance(block, (ProcessBlock, MultiProcessBlock)):
                resources = []
                
                if isinstance(block, ProcessBlock):
                    # Skip if no resource
                    if block.resource is not None:
                        resources = [block.resource]
                else:
                    resources = list(block.resource_requirements.keys())
                
                for resource in resources:
                    resource_name = self._find_resource_name(resource)
                    
                    if resource_name:
                        # Check if resource type matches usage
                        if isinstance(resource, simpy.PriorityResource):
                            # Priority resource should be used with priority entities
                            if not self._has_priority_generator():
                                self.warnings.append(
                                    f"PRIORITY MISMATCH: Resource '{resource_name}' is "
                                    f"PriorityResource but no entities have priorities. "
                                    f"Consider using regular Resource instead."
                                )
                        
                        # Check for PreemptiveResource (if supported)
                        if isinstance(resource, simpy.PreemptiveResource):
                            self.warnings.append(
                                f"PREEMPTIVE RESOURCE: '{resource_name}' is PreemptiveResource. "
                                f"Ensure your code handles preemption correctly."
                            )
    
    def _validate_multi_resource_blocks(self):
        """Validate blocks that require multiple resources simultaneously."""
        from desk.blocks.process_block import MultiProcessBlock
        
        for block_name, block in self.model.blocks.items():
            if isinstance(block, MultiProcessBlock):
                total_units = sum(block.resource_requirements.values())
                
                # WARNING: Requesting many resources
                if total_units > 5:
                    self.warnings.append(
                        f"COMPLEX RESOURCE REQUIREMENTS: Block '{block_name}' "
                        f"requires {total_units} total resource units across "
                        f"{len(block.resource_requirements)} resources. "
                        f"This may increase chance of deadlock."
                    )
                
                # Check for potential deadlock with other multi-resource blocks
                self._check_circular_dependencies(block_name, block)
    
    def _check_blocking_deadlock_cycles(self):
        """
        Lightweight structural deadlock check for the blocking-after-service
        hand-off mechanism (ProcessBlock/MultiProcessBlock.request_admission).

        Background: a ProcessBlock/MultiProcessBlock with blocking=True
        (the default) does NOT release its resource(s) the instant its own
        service finishes -- it first waits for the entity to be admitted
        into whatever comes next (following through any DecideBlock
        routing, and through resource-less/blocking=False blocks, which
        are transparent). Only then does it release.

        That creates a "waits-for" relationship: block X depends on block
        Y admitting its entity before X can free its resource(s). Walking
        the block graph and following that relationship for every
        blocking, capacity-holding block builds a directed graph. If that
        graph contains a cycle -- X depends on Y depends on ... depends on
        X -- then a circular wait is structurally possible: if every
        resource along the cycle happens to be fully occupied at the same
        time, each block is stuck holding what it has, waiting for
        capacity that will only be freed by another block that is itself
        stuck waiting on the first one. The simulation hangs (entities
        stop advancing) rather than crashing, which makes it worth
        catching here rather than discovering it as a stalled run.

        This is a STATIC/STRUCTURAL check on the block graph, not a proof
        that a deadlock will actually happen at runtime -- it depends on
        which DecideBlock routes are actually taken and on timing/capacity.
        Findings are reported as WARNINGS, not errors, so they don't block
        `raise_on_error=True` runs; they're a strong hint to review the
        routing or, if the cycle is truly benign in practice, silence it
        deliberately with `blocking=False` on one edge of the cycle.
        """
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        from desk.blocks.decide_block import DecideBlock

        def is_capacity_block(block) -> bool:
            """A block that actually holds a resource and won't release it
            until downstream admits the entity (blocking=True, the default)."""
            if isinstance(block, ProcessBlock):
                return block.resource is not None and getattr(block, 'blocking', True)
            if isinstance(block, MultiProcessBlock):
                return len(block.resource_requirements) > 0 and getattr(block, 'blocking', True)
            return False

        capacity_blocks = {
            name: block for name, block in self.model.blocks.items()
            if is_capacity_block(block)
        }

        def next_capacity_blocks(start_block) -> set:
            """
            Find every capacity-holding block reachable from start_block's
            next_block without crossing another capacity-holding,
            blocking=True block first -- crossing one of those IS the
            dependency edge, so the search stops there. DecideBlock fans
            out into every one of its configured routes (a block waiting
            on a DecideBlock is really waiting on whichever concrete
            destination gets chosen). A local `seen` set guards against
            infinite recursion if the block graph itself has a cycle
            (e.g. two DecideBlocks routing back and forth) that never
            passes through a capacity block.
            """
            found = set()
            seen = set()

            def walk(block):
                if block is None or id(block) in seen:
                    return
                seen.add(id(block))

                if isinstance(block, DecideBlock):
                    for route in block.routes.values():
                        walk(route.get('block'))
                    return

                if is_capacity_block(block) and block is not start_block:
                    found.add(block.name)
                    return  # this is the dependency -- don't look past it

                # Transparent hop: resource-less ProcessBlock, a
                # blocking=False capacity block (no wait created, so
                # keep walking past it), DisposeBlock, or a custom block.
                walk(getattr(block, 'next_block', None))

            walk(getattr(start_block, 'next_block', None))
            return found

        waits_for = {
            name: next_capacity_blocks(block)
            for name, block in capacity_blocks.items()
        }

        # Standard white/gray/black DFS cycle detection. Reports one
        # representative cycle per loop found -- lightweight by design,
        # not an exhaustive enumeration of every path through it.
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in waits_for}
        path_stack = []
        reported = set()

        def dfs(node):
            color[node] = GRAY
            path_stack.append(node)
            for neighbor in waits_for.get(node, ()):
                if neighbor not in waits_for:
                    continue  # leads to a non-blocking block: no wait-for edge
                if color[neighbor] == GRAY:
                    cycle_start = path_stack.index(neighbor)
                    cycle = path_stack[cycle_start:] + [neighbor]
                    key = tuple(sorted(set(cycle)))
                    if key not in reported:
                        reported.add(key)
                        self.warnings.append(
                            "POTENTIAL DEADLOCK CYCLE: " + " -> ".join(cycle) +
                            ". Each block here holds its resource until the "
                            "next one admits the entity, and the last one "
                            "waits on the first. If every resource along this "
                            "path fills up at once, the whole cycle can hang "
                            "forever. Review capacities/routing on this path, "
                            "or set blocking=False on one link to break the cycle."
                        )
                elif color[neighbor] == WHITE:
                    dfs(neighbor)
            path_stack.pop()
            color[node] = BLACK

        for name in waits_for:
            if color[name] == WHITE:
                dfs(name)

    def _check_circular_dependencies(self, block_name: str, block):
        """
        Check for potential circular dependencies in resource requirements.
        
        This is a simplified check - full deadlock detection is complex.
        """
        from desk.blocks.process_block import MultiProcessBlock
        
        block_resources = set(block.resource_requirements.keys())
        
        for other_name, other_block in self.model.blocks.items():
            if other_name == block_name:
                continue
            
            if isinstance(other_block, MultiProcessBlock):
                other_resources = set(other_block.resource_requirements.keys())
                
                # If blocks share resources, potential for deadlock
                if block_resources.intersection(other_resources):
                    self.warnings.append(
                        f"SHARED RESOURCES: Blocks '{block_name}' and '{other_name}' "
                        f"both require some of the same resources. This could lead to "
                        f"deadlock if not carefully designed. Review the model logic."
                    )
                    break  # Only warn once per block
    
    def _has_priority_generator(self) -> bool:
        """Check if any CreateBlock has priority generator."""
        from desk.blocks.create_block import CreateBlock
        
        for block in self.model.blocks.values():
            if isinstance(block, CreateBlock):
                if block.priority_generator is not None:
                    return True
        return False
    
    def _find_resource_name(self, resource_obj) -> Optional[str]:
        """Find resource name from resource object."""
        if resource_obj is None:
            return None
        
        for name, res in self.model.resources.items():
            if res == resource_obj:
                return name
        return None
    
    def _print_validation_results(self):
        """Print validation results with color coding."""
        if not self.errors and not self.warnings:
            print("\n" + "=" * 70)
            print("RESOURCE VALIDATION: ALL CHECKS PASSED")
            print("=" * 70)
            return
        
        print("\n" + "=" * 70)
        print("RESOURCE VALIDATION RESULTS")
        print("=" * 70)
        
        if self.errors:
            print(f"\nCRITICAL ERRORS FOUND: {len(self.errors)}")
            print("-" * 70)
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")
        
        if self.warnings:
            print(f"\nWARNINGS: {len(self.warnings)}")
            print("-" * 70)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")
        
        print("=" * 70)
    
    def _format_error_message(self) -> str:
        """Format errors into exception message."""
        msg = f"\n{len(self.errors)} CRITICAL RESOURCE CONFIGURATION ERROR(S) FOUND:\n\n"
        for i, error in enumerate(self.errors, 1):
            msg += f"{i}. {error}\n"
        msg += "\nFIX THESE ERRORS BEFORE RUNNING SIMULATION!"
        return msg
    
    def print_resource_summary(self):
        """Print summary of all resources and their usage."""
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        print("\n" + "=" * 70)
        print("RESOURCE CONFIGURATION SUMMARY")
        print("=" * 70)
        
        # Count resource-free blocks
        delay_only_blocks = []
        
        for name, resource in sorted(self.model.resources.items()):
            capacity = resource.capacity
            resource_type = self._get_resource_type_name(resource)
            
            print(f"\nResource: {name}")
            print(f"  Type: {resource_type}")
            print(f"  Capacity: {capacity} units")
            
            # Find blocks using this resource
            using_blocks = []
            total_max_usage = 0
            
            for block_name, block in self.model.blocks.items():
                if isinstance(block, ProcessBlock):
                    # Track delay-only blocks separately
                    if block.resource is None:
                        delay_only_blocks.append(block_name)
                    elif block.resource == resource:
                        units = getattr(block, 'resource_units', 1)
                        using_blocks.append((block_name, units))
                        total_max_usage = max(total_max_usage, units)
                
                elif isinstance(block, MultiProcessBlock):
                    if resource in block.resource_requirements:
                        units = block.resource_requirements[resource]
                        using_blocks.append((block_name, units))
                        total_max_usage = max(total_max_usage, units)
            
            if using_blocks:
                print(f"  Used by {len(using_blocks)} block(s):")
                for block_name, units in using_blocks:
                    pct = (units / capacity * 100) if capacity > 0 else 0
                    print(f"    - {block_name}: {units} units ({pct:.0f}% of capacity)")
                
                print(f"  Maximum single allocation: {total_max_usage} units "
                      f"({total_max_usage/capacity*100:.0f}% of capacity)")
            else:
                print(f"  WARNING: Resource not used by any block!")
        
        # Print delay-only blocks summary
        if delay_only_blocks:
            print("\n" + "-" * 70)
            print("DELAY-ONLY BLOCKS (No Resource Required):")
            print(f"  Found {len(delay_only_blocks)} pure delay block(s):")
            for block_name in delay_only_blocks:
                print(f"    - {block_name} (pure delay operation)")
            print("  Note: These blocks perform time delays without consuming resources.")
        
        print("=" * 70)
    
    def _get_resource_type_name(self, resource) -> str:
        """Get human-readable resource type name."""
        if isinstance(resource, simpy.PreemptiveResource):
            return "PreemptiveResource"
        elif isinstance(resource, simpy.PriorityResource):
            return "PriorityResource"
        elif isinstance(resource, simpy.Resource):
            return "Resource"
        else:
            return "Unknown"