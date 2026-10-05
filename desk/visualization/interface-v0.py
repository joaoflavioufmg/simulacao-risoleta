# =============================================================================
# FILE: visualization/generic_visualizer.py
# =============================================================================
"""
Generic real-time visualization interface for simulation models.

FIXES:
1. Connectors now properly exit/enter blocks from outside (not inside)
2. Entities flow smoothly along connector paths
3. Queue statistics now match visual queue counts

(USER FIX) 4. Play/Step buttons now correctly drive the simulation and animation
           incrementally, instead of running to completion.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from desk.blocks.create_block import CreateBlock


# =============================================================================
# Event System for Communication Between Simulation and GUI
# =============================================================================
@dataclass
class VisualizationEvent:
    """Event sent from simulation to GUI."""
    event_type: str  # 'entity_created', 'entity_moved', 'entity_disposed', 'stats_update'
    timestamp: float
    data: Dict[str, Any]


class EventQueue:
    """Thread-safe queue for passing events from simulation to GUI."""
    def __init__(self):
        self.queue = queue.Queue()
    
    def put(self, event: VisualizationEvent):
        """Add event to queue."""
        self.queue.put(event)
    
    def get_all(self) -> List[VisualizationEvent]:
        """Get all pending events."""
        events = []
        while not self.queue.empty():
            try:
                events.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return events


# =============================================================================
# Model Inspector - Extracts Structure from Simulation Model
# =============================================================================
class ModelInspector:
    """Extracts structural information from simulation model."""
    
    @staticmethod
    def extract_structure(model) -> Dict[str, Any]:
        """
        Extract block information and connections from model.
        
        Returns:
            Dictionary with:
            - blocks: List of block names and types
            - connections: List of (from_block, to_block) tuples
            - resources: Dictionary of resource names and capacities
        """
        from desk.blocks.create_block import CreateBlock
        from desk.blocks.dispose_block import DisposeBlock
        from desk.blocks.decide_block import DecideBlock
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        structure = {
            'blocks': {},
            'connections': [],
            'resources': {}
        }
        
        # Extract blocks
        for name, block in model.blocks.items():
            block_info = {
                'name': name,
                'type': type(block).__name__,
                'is_source': isinstance(block, CreateBlock),
                'is_sink': isinstance(block, DisposeBlock),
                'is_decision': isinstance(block, DecideBlock),
                'is_process': isinstance(block, (ProcessBlock, MultiProcessBlock))
            }
            structure['blocks'][name] = block_info
        
        # Extract connections (regular)
        for name, block in model.blocks.items():
            if block.next_block:
                structure['connections'].append((name, block.next_block.name))
        
        # Extract decision routes
        for name, block in model.blocks.items():
            if isinstance(block, DecideBlock):
                for route_name, route_info in block.routes.items():
                    target_block = route_info['block']
                    structure['connections'].append((name, target_block.name))
        
        # Extract resources
        for res_name, resource in model.resources.items():
            structure['resources'][res_name] = {
                'capacity': resource.capacity,
                'type': type(resource).__name__
            }
        
        return structure


# =============================================================================
# Auto-Layout Generator
# =============================================================================
class AutoLayout:
    """Automatically generates layout positions for blocks with overlap prevention."""

    # Minimum pixel gaps between block bounding boxes
    MIN_H_GAP = 30   # horizontal gap between blocks in the same level
    MIN_V_GAP = 20   # vertical gap between blocks in different levels
    BLOCK_W   = 130  # nominal block width (same as _draw_blocks uses ≈120, +margin)
    BLOCK_H   = 50   # nominal block height

    @staticmethod
    def generate(structure: Dict[str, Any],
                 canvas_width: int = 1000,
                 canvas_height: int = 600) -> Dict[str, Tuple[int, int]]:
        """
        Generate automatic layout.
        1. Assign levels (BFS from sources).
        2. Compute required canvas size so blocks never overlap.
        3. Position blocks with proper spacing inside each level.
        4. Apply a vertical force-directed pass to further separate nodes.
        Returns:
            Dict mapping block_name -> (x, y) centre coordinates.
        """
        blocks      = structure['blocks']
        connections = structure['connections']

        graph = {name: [] for name in blocks.keys()}
        for from_block, to_block in connections:
            graph[from_block].append(to_block)

        sources = [name for name, info in blocks.items() if info['is_source']]
        levels  = AutoLayout._assign_levels(graph, sources)

        # --- compute required canvas extents ----------------------------
        level_groups: Dict[int, list] = {}
        for node, lvl in levels.items():
            level_groups.setdefault(lvl, []).append(node)

        max_level    = max(levels.values()) if levels else 0
        max_in_level = max(len(v) for v in level_groups.values()) if level_groups else 1

        # Required width: enough room for (max_level+1) columns
        col_w = AutoLayout.BLOCK_W + AutoLayout.MIN_H_GAP
        req_w = max(canvas_width, col_w * (max_level + 2) + 100)

        # Required height: enough room for the tallest column
        row_h = AutoLayout.BLOCK_H + AutoLayout.MIN_V_GAP
        req_h = max(canvas_height, row_h * max_in_level + 120)

        positions = AutoLayout._calculate_positions(levels, level_groups,
                                                     max_level, req_w, req_h)
        positions = AutoLayout._force_spread(positions, req_w, req_h)
        return positions

    @staticmethod
    def _assign_levels(graph: Dict[str, List[str]],
                       sources: List[str]) -> Dict[str, int]:
        """Assign BFS depth levels, ensuring max-depth wins for shared nodes."""
        levels:  Dict[str, int] = {}
        visited: set = set()
        queue_bfs = [(source, 0) for source in sources]

        while queue_bfs:
            node, level = queue_bfs.pop(0)
            if node in visited:
                continue
            visited.add(node)
            levels[node] = level
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue_bfs.append((neighbor, level + 1))

        for node in graph.keys():
            if node not in levels:
                levels[node] = 0
        return levels

    @staticmethod
    def _calculate_positions(levels, level_groups, max_level,
                              width, height) -> Dict[str, Tuple[int, int]]:
        """Place blocks in a clean grid: columns = levels, rows = nodes per level."""
        margin_x = 80
        margin_y = 80
        usable_w  = width  - 2 * margin_x
        usable_h  = height - 2 * margin_y

        col_step = usable_w / (max_level + 1) if max_level > 0 else usable_w

        positions: Dict[str, Tuple[int, int]] = {}
        for level, nodes in level_groups.items():
            x = int(margin_x + level * col_step)
            n = len(nodes)
            if n == 1:
                y_list = [height // 2]
            else:
                row_step = usable_h / (n - 1)
                y_list = [int(margin_y + i * row_step) for i in range(n)]
            for node, y in zip(nodes, y_list):
                positions[node] = (x, y)
        return positions

    @staticmethod
    def _force_spread(positions: Dict[str, Tuple[int, int]],
                      width: int, height: int,
                      iterations: int = 120) -> Dict[str, Tuple[int, int]]:
        """
        Force-directed vertical (+ mild horizontal) spreading so blocks
        never visually overlap.  X-axis movement is intentionally weak to
        preserve the left-to-right flow order.
        """
        BW = AutoLayout.BLOCK_W
        BH = AutoLayout.BLOCK_H
        REPULSE  = 8000.0
        DAMPING  = 0.80
        X_WEIGHT = 0.15   # keep horizontal order intact

        nodes = list(positions.keys())
        pos = {n: [float(positions[n][0]), float(positions[n][1])] for n in nodes}
        vel = {n: [0.0, 0.0] for n in nodes}
        # Anchor x-positions (we only nudge y significantly)
        anchor_x = {n: pos[n][0] for n in nodes}

        for _ in range(iterations):
            forces = {n: [0.0, 0.0] for n in nodes}

            for i, a in enumerate(nodes):
                for b in nodes[i + 1:]:
                    dx = pos[b][0] - pos[a][0]
                    dy = pos[b][1] - pos[a][1]
                    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)

                    # Desired clearance based on block dimensions
                    desired = ((BW + 20) ** 2 + (BH + 20) ** 2) ** 0.5
                    if dist < desired:
                        f  = REPULSE / (dist ** 2)
                        fx = f * dx / dist
                        fy = f * dy / dist
                        forces[a][0] -= fx
                        forces[a][1] -= fy
                        forces[b][0] += fx
                        forces[b][1] += fy

            for n in nodes:
                # Weak spring pulling back to anchor x
                spring_x = (anchor_x[n] - pos[n][0]) * 2.0
                vel[n][0] = (vel[n][0] + forces[n][0] * X_WEIGHT + spring_x) * DAMPING
                vel[n][1] = (vel[n][1] + forces[n][1]) * DAMPING
                pos[n][0] = max(80, min(width  - 80, pos[n][0] + vel[n][0]))
                pos[n][1] = max(80, min(height - 80, pos[n][1] + vel[n][1]))

        return {n: (int(pos[n][0]), int(pos[n][1])) for n in nodes}


# =============================================================================
# Main Visualization GUI
# =============================================================================
class SimulationVisualizer:
    """
    Generic real-time visualization for simulation models.
    
    Usage:
        visualizer = SimulationVisualizer(model_builder)
        visualizer.run()  # Starts GUI in main thread
    """
    
    def __init__(self, model_builder, 
                 canvas_width: int = 1000,
                 canvas_height: int = 600,
                 custom_positions: Optional[Dict[str, Tuple[int, int]]] = None):
        """
        Initialize visualizer.
        
        Args:
            model_builder: Function that returns a new model instance
            canvas_width: Canvas width in pixels
            canvas_height: Canvas height in pixels
            custom_positions: Optional manual positions for blocks
        """
        self.model_builder = model_builder
        self.model = self.model_builder()
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Extract model structure and generate layout
        self.structure = ModelInspector.extract_structure(self.model)
        if custom_positions:
            self.positions = custom_positions
        else:
            self.positions = AutoLayout.generate(
                self.structure, canvas_width, canvas_height
            )
        
        # Event queue
        self.event_queue = EventQueue()
        
        # Instrument model
        self.instrument = VisualizationInstrument(self.model, self.event_queue)
        
        # GUI state
        self.root = None
        self.canvas = None
        self.entities_on_canvas = {}  # entity_id -> (circle, text)
        self.block_widgets = {}  # block_name -> widget_ids
        self.stats_labels = {}

        # Visualization state
        self.connection_paths = {}  # (from, to) -> [(x, y), ...]  (logical, kept for fallback)
        self.queue_areas = {}       # block_name -> (x1, y1, x2, y2)  (logical)
        self.block_centers = {}     # block_name -> (x, y)             (logical)
        self.entity_queue_slots = {}# block_name -> [entity_id, ...]
        self.service_areas = {}     # block_name -> (x1, y1, x2, y2)  (logical)
        self.entity_service_slots = {}# block_name -> [entity_id, ...]
        self.resource_to_blocks_map = {} # Maps res_name -> [block_name]

        # Canvas item IDs — used to read CURRENT (zoomed) positions from canvas
        self.queue_rect_ids   = {}  # block_name -> canvas id of dashed queue rect
        self.service_rect_ids = {}  # block_name -> canvas id of service rect (shape)
        self.connection_line_ids = {}  # (from, to) -> canvas id of arrow line
        
        # Map resources to blocks
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        for res_name, res_obj in self.model.resources.items():
            self.resource_to_blocks_map[res_name] = []
            for block_name, block in self.model.blocks.items():
                if isinstance(block, ProcessBlock) and block.resource == res_obj:
                    self.resource_to_blocks_map[res_name].append(block_name)
                elif isinstance(block, MultiProcessBlock) and res_obj in block.resource_requirements:
                     self.resource_to_blocks_map[res_name].append(block_name)
        
        # Statistics tracking
        self.stats = {
            'total_created': 0,
            'total_disposed': 0,
            'current_wip': 0,
            'simulation_time': 0.0
        }
        
        # Animation settings
        self.animation_speed = 0.02  # seconds per step
        self.steps_per_move = 20
        
        # Playback control
        self.is_paused = True
        self.is_running = False
        self.speed_multiplier = 1.0
        
        # Control widgets
        self.play_button = None
        self.speed_label = None
        self.progress_bar = None
        self.step_pause_timer = None

        # Generation counter: incremented on every reset so stale animation
        # callbacks silently exit instead of moving already-deleted canvas items.
        self._animation_generation = 0
        
        # (1) ADD: Simulation time limit (will be set by run())
        self._simulation_time_limit = float('inf')

    def setup_gui(self):
        """Setup tkinter GUI components."""
        self.root = tk.Tk()
        self.root.title("Simulation Visualizer")
        
        # Main container
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.Frame(container, relief=tk.RAISED, borderwidth=2)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self._create_control_panel(control_frame)
        
        # Main content
        main_frame = ttk.Frame(container)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas frame with scrollbars (needed for large auto-layouts)
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = ZoomableCanvas(
            canvas_frame,
            on_zoom=self._on_zoom_changed,   # ← callback for font/entity rescaling
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)
        
        
        # Stats panel — wider to accommodate Block Types legend        
        stats_frame = ttk.Frame(main_frame, width=270)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        stats_frame.pack_propagate(False)
        
        ttk.Label(stats_frame, text="Statistics", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        self._create_stats_panel(stats_frame)
        
        # Draw initial structure
        self._draw_blocks()
        self._draw_connections()
        
        # Setup shortcuts
        self._setup_keyboard_shortcuts()

        # Update scroll region to match actual content size
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Animate blocks into their auto-layout positions (yEd-style)
        self.root.after(150, lambda: self._animate_layout_to_positions(self.positions))
        
        # Start event processing AND simulation tick
        self.root.after(50, self._process_events)
        self.root.after(50, self._simulation_tick)
    
    def _create_control_panel(self, parent):
        """Create playback control panel."""
        # Title
        title_label = ttk.Label(parent, text="▶ Simulation Controls", 
                               font=("Arial", 11, "bold"))
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Play/Pause button
        self.play_button = ttk.Button(
            parent, text="▶ Play", 
            command=self._toggle_play_pause,
            width=10
        )
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        ttk.Button(
            parent, text="⟲ Reset",
            command=self._reset_simulation,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Step forward button
        ttk.Button(
            parent, text="⏭ Step",
            command=self._step_forward,
            width=8
        ).pack(side=tk.LEFT, padx=5)
        
        # Speed controls
        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )
        
        ttk.Label(parent, text="Speed:", font=("Arial", 10)).pack(
            side=tk.LEFT, padx=5
        )
        
        # Speed preset buttons
        speed_frame = ttk.Frame(parent)
        speed_frame.pack(side=tk.LEFT)
        
        speeds = [
            ("0.25x", 0.25),
            ("0.5x", 0.5),
            ("1x", 1.0),
            ("2x", 2.0),
            ("5x", 5.0),
            ("10x", 10.0),
            ("MAX", 50.0)
        ]
        
        for label, speed in speeds:
            btn = ttk.Button(
                speed_frame, text=label,
                command=lambda s=speed: self._set_speed(s),
                width=6
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # Current speed display
        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )
        
        self.speed_label = ttk.Label(
            parent, text="Current: 1.0x",
            font=("Arial", 10, "bold"),
            foreground="blue"
        )
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        # Progress indicator
        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )
        
        status_frame = ttk.Frame(parent)
        status_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(status_frame, text="Status:", 
                 font=("Arial", 9)).pack(anchor=tk.W)
        
        self.status_label = ttk.Label(
            status_frame, text="Ready",
            font=("Arial", 9, "bold"),
            foreground="green"
        )
        self.status_label.pack(anchor=tk.W)

        # ── Zoom controls ─────────────────────────────────────────────────
        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )
        zoom_frame = ttk.Frame(parent)
        zoom_frame.pack(side=tk.LEFT, padx=2)
        ttk.Label(zoom_frame, text="Zoom:", font=("Arial", 9)).pack(anchor=tk.W)
        zoom_btn_row = ttk.Frame(zoom_frame)
        zoom_btn_row.pack()
        ttk.Button(
            zoom_btn_row, text="⟲ 1:1",
            command=self._reset_zoom,
            width=7
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            zoom_btn_row, text="⤢ Fit",
            command=self._fit_view,
            width=7
        ).pack(side=tk.LEFT, padx=2)

    def _create_stats_panel(self, parent):
        """Create statistics display panel."""
        # General stats
        ttk.Label(parent, text="Simulation Time:", font=("Arial", 10)).pack(anchor=tk.W)
        self.stats_labels['simulation_time'] = ttk.Label(parent, text="0.0", font=("Arial", 10))
        self.stats_labels['simulation_time'].pack(anchor=tk.W)

        ttk.Label(parent, text="Entities Created:", font=("Arial", 10)).pack(anchor=tk.W)
        self.stats_labels['total_created'] = ttk.Label(parent, text="0", font=("Arial", 10))
        self.stats_labels['total_created'].pack(anchor=tk.W)

        ttk.Label(parent, text="Entities Disposed:", font=("Arial", 10)).pack(anchor=tk.W)
        self.stats_labels['total_disposed'] = ttk.Label(parent, text="0", font=("Arial", 10))
        self.stats_labels['total_disposed'].pack(anchor=tk.W)

        ttk.Label(parent, text="Current WIP:", font=("Arial", 10)).pack(anchor=tk.W)
        self.stats_labels['current_wip'] = ttk.Label(parent, text="0", font=("Arial", 10))
        self.stats_labels['current_wip'].pack(anchor=tk.W)

        # Resources section
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="Resources", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        
        for res_name in sorted(self.structure['resources'].keys()):
            resource = self.model.resources[res_name]
            capacity = resource.capacity  # <-- directly from simpy.Resource

            res_frame = ttk.Frame(parent)
            res_frame.pack(fill=tk.X, pady=2)

            # Shows "4 doctors:" or "2 nurses:"
            ttk.Label(res_frame, text=f"{capacity} {res_name}:", width=14).pack(side=tk.LEFT)

            ttk.Label(res_frame, text="Util:").pack(side=tk.LEFT, padx=2)
            util_key = f"{res_name}_util"
            self.stats_labels[util_key] = ttk.Label(res_frame, text="0.00%", width=8)
            self.stats_labels[util_key].pack(side=tk.LEFT)

            ttk.Label(res_frame, text="Queue:").pack(side=tk.LEFT, padx=2)
            queue_key = f"{res_name}_queue"
            self.stats_labels[queue_key] = ttk.Label(res_frame, text="0", width=5)
            self.stats_labels[queue_key].pack(side=tk.LEFT)

        # Block Types legend at the bottom of the panel
        self._draw_legend_panel(parent)


    # ------------------------------------------------------------------
    # Zoom-aware canvas coordinate helpers
    # ------------------------------------------------------------------
    def _current_block_center(self, name):
        """Return the CURRENT (post-zoom) canvas centre of a block."""
        if name not in self.block_widgets:
            return self.block_centers.get(name, (0, 0))
        shape, _ = self.block_widgets[name]
        try:
            bbox = self.canvas.bbox(shape)
            if bbox:
                return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        except tk.TclError:
            pass
        return self.block_centers.get(name, (0, 0))

    def _current_queue_area(self, name):
        """Return the CURRENT (post-zoom) bounding box of the queue rectangle."""
        rid = self.queue_rect_ids.get(name)
        if rid:
            try:
                bbox = self.canvas.bbox(rid)
                if bbox:
                    return bbox   # (x1, y1, x2, y2)
            except tk.TclError:
                pass
        return self.queue_areas.get(name)

    def _current_service_area(self, name):
        """Return the CURRENT (post-zoom) bounding box of the service/block rect."""
        rid = self.service_rect_ids.get(name)
        if rid:
            try:
                bbox = self.canvas.bbox(rid)
                if bbox:
                    return bbox
            except tk.TclError:
                pass
        return self.service_areas.get(name)

    def _current_connection_path(self, from_name, to_name, num_points: int = 20):
        """
        Compute a path between two blocks using their CURRENT canvas positions.
        Reads the actual arrow-line endpoints from the canvas item so the path
        is correct at any zoom level.
        """
        lid = self.connection_line_ids.get((from_name, to_name))
        if lid:
            try:
                coords = self.canvas.coords(lid)  # [x1, y1, x2, y2]
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[0], coords[1], coords[-2], coords[-1]
                    return [(x1 + (x2 - x1) * i / num_points,
                             y1 + (y2 - y1) * i / num_points)
                            for i in range(num_points + 1)]
            except tk.TclError:
                pass
        # Fallback: compute from current block centres
        fx, fy = self._current_block_center(from_name)
        tx, ty = self._current_block_center(to_name)
        return [(fx + (tx - fx) * i / num_points,
                 fy + (ty - fy) * i / num_points)
                for i in range(num_points + 1)]

    # ------------------------------------------------------------------
    def _draw_blocks(self):
        """Draw all blocks on canvas and store canvas item IDs."""
        block_width = 120
        block_height = 40
        for name, (x, y) in self.positions.items():
            info = self.structure['blocks'][name]
            if info['is_source']:
                color = "lightgreen"
            elif info['is_sink']:
                color = "lightpink"
            elif info['is_decision']:
                color = "lightyellow"
            else:
                color = "lightblue"
            x1 = x - block_width / 2
            y1 = y - block_height / 2
            x2 = x + block_width / 2
            y2 = y + block_height / 2

            # Draw diamond for DECIDE blocks
            if info['is_decision']:
                diamond_points = [
                    x, y - 35,
                    x + 60, y,
                    x, y + 35,
                    x - 60, y
                ]
                shape = self.canvas.create_polygon(
                    diamond_points, fill=color, outline="black", width=2
                )
                text = self.canvas.create_text(x, y, text=name, font=("Arial", 9, "bold"))
                self.block_widgets[name] = (shape, text)
                self.block_centers[name] = (x, y)
                self.service_rect_ids[name] = shape  # polygon serves as shape ID

            # Default rectangle for others
            else:
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
                text = self.canvas.create_text(x, y, text=name, width=block_width - 10, justify=tk.CENTER)
                self.block_widgets[name] = (rect, text)
                self.block_centers[name] = (x, y)
                self.service_rect_ids[name] = rect  # store shape ID

            if info['is_process']:
                # Queue area above the block
                q_y1 = y1 - block_height
                q_y2 = y1
                self.queue_areas[name] = (x1, q_y1, x2, q_y2)
                qrid = self.canvas.create_rectangle(
                    self.queue_areas[name], dash=(2, 2), fill="white"
                )
                self.queue_rect_ids[name] = qrid   # ← store canvas ID
                self.entity_queue_slots.setdefault(name, [])
                # Service area = block rectangle
                self.service_areas[name] = (x1, y1, x2, y2)
                self.entity_service_slots.setdefault(name, [])

    def _draw_connections(self):
        """Draw connections between blocks and store canvas line IDs."""
        for from_name, to_name in self.structure['connections']:
            from_pos = self.positions[from_name]
            to_pos   = self.positions[to_name]
            x1 = from_pos[0] + 60   # right edge of from-block
            y1 = from_pos[1]
            x2 = to_pos[0]   - 60   # left  edge of to-block
            y2 = to_pos[1]
            lid = self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=2)
            self.connection_line_ids[(from_name, to_name)] = lid   # ← store ID
            # Logical path (kept for fallback / non-zoomed use)
            num_points = 20
            self.connection_paths[(from_name, to_name)] = [
                (x1 + (x2 - x1) * i / num_points,
                 y1 + (y2 - y1) * i / num_points)
                for i in range(num_points + 1)
            ]

    def _draw_legend_panel(self, parent):
        """Draw Block Types legend inside the right stats panel (not on canvas)."""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="Block Types", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        items = [
            ("CREATE  (Source)",    "#90EE90"),   # lightgreen
            ("DISPOSE (Sink)",      "#FFB6C1"),   # lightpink
            ("DECIDE  (Decision) ◆","#FFFFE0"),   # lightyellow
            ("PROCESS (Activity)",  "#ADD8E6"),   # lightblue
        ]
        for label_text, hex_color in items:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=1)
            # Colour swatch drawn on a tiny canvas
            swatch = tk.Canvas(row, width=18, height=14,
                               bg=hex_color, highlightthickness=1,
                               highlightbackground="black")
            swatch.pack(side=tk.LEFT, padx=(2, 5))
            ttk.Label(row, text=label_text, font=("Arial", 9)).pack(side=tk.LEFT)

    def _process_events(self):
        """Process pending visualization events."""
        events = self.event_queue.get_all()
        for event in events:
            if event.event_type == 'entity_created':
                self._handle_entity_created(event)
            elif event.event_type == 'entity_moved':
                self._handle_entity_moved(event)
            elif event.event_type == 'entity_disposed':
                self._handle_entity_disposed(event)
            elif event.event_type == 'stats_update':
                self._handle_stats_update(event)
        
        # (3) MODIFY: Reschedule itself
        self.root.after(50, self._process_events)

    def _handle_entity_created(self, event):
        """Handle entity creation — size and font are scaled to current zoom."""
        data          = event.data
        entity_id     = data['entity_id']
        entity_number = data['entity_number']
        block_name    = data['block_name']
        x, y = self._current_block_center(block_name)

        # Scale radius and font to match current zoom level
        scale  = getattr(self.canvas, '_scale', 1.0)
        r      = max(5, self._BASE_ENTITY_RADIUS * scale)
        efont  = ("Arial", max(4, int(self._BASE_ENTITY_FONT * scale)), "bold")

        circle = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")
        text   = self.canvas.create_text(x, y - 1, text=str(entity_number),
                                         fill="white", font=efont)
        self.entities_on_canvas[entity_id] = (circle, text)
        self.stats['total_created'] += 1
        self.stats['current_wip']   += 1
        self._update_stats_display()

    def _handle_entity_moved(self, event):
        """Handle entity moved event."""
        data = event.data
        entity_id = data['entity_id']
        from_block = data['from_block']
        to_block = data['to_block']
        state = data['state']
        if entity_id not in self.entities_on_canvas:
            return
        circle, text = self.entities_on_canvas[entity_id]
        self._animate_move_along_path(entity_id, circle, text, from_block, to_block, state)

    def _handle_entity_disposed(self, event):
        """Handle entity disposed event."""
        entity_id = event.data['entity_id']
        if entity_id in self.entities_on_canvas:
            circle, text = self.entities_on_canvas[entity_id]
            self.canvas.delete(circle)
            self.canvas.delete(text)
            del self.entities_on_canvas[entity_id]
        self.stats['total_disposed'] += 1
        self.stats['current_wip'] -= 1
        self._update_stats_display()


    # New function to initialize SimPy generators
    def _initialize_simulation(self):
        """Initializes the SimPy generators."""
        try:
            for block in self.model.blocks.values():
                if isinstance(block, CreateBlock):
                    self.model.env.process(block._generation_process())
            self.is_running = True # Mark as "ready to run"
            self.is_paused = True # Start paused
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            self.is_running = False

    #  New function to drive the simulation from the GUI thread
    def _simulation_tick(self):
        """Advances the simulation by one step or time interval."""
        
        # 1. Check if simulation is running and not paused
        if not self.is_running or self.is_paused:
            # If paused or stopped, just check again later
            self.root.after(100, self._simulation_tick) # Check again in 100ms
            return

        # 2. Check if simulation is complete
        # (Compare against sim time limit OR check if events are exhausted)
        next_event_time = self.model.env.peek()
        is_complete = (self.model.env.now >= self._simulation_time_limit) or \
                      (next_event_time == float('inf'))
                      
        if is_complete:
            self.is_running = False
            self.is_paused = True
            self.play_button.config(text="▶ Play")
            self.status_label.config(text="Completed", foreground="green")
            self.root.after(100, self._simulation_tick) # Keep the loop alive but inactive
            return

        
        # The 'delay_ms' for the GUI update is based on animation_speed
        delay_ms = max(1, int(self.animation_speed * 1000)) 
        
        # How much *simulation time* per tick        
        # This is proportional to the speed multiplier.
        # 1 tick = 0.1 sim time units at 1x speed.
        sim_step = 0.1 * self.speed_multiplier
        
        # If speed is MAX (50.0), run for a larger chunk.
        if self.speed_multiplier >= 50.0:
            sim_step = 5.0 * self.speed_multiplier # Run much faster
            delay_ms = 1 # Update GUI as fast as possible

        run_until = self.model.env.now + sim_step
        
        # Don't run past the end time
        run_until = min(run_until, self._simulation_time_limit)
        
        # ... but also don't run past the next scheduled event if we are running slowly
        if self.speed_multiplier < 5.0:
             run_until = min(run_until, next_event_time + 0.00001)

        # Run the simulation for that interval
        try:
            self.model.env.run(until=run_until)
        except Exception as e:
            # Catch simulation errors
            messagebox.showerror("Simulation Error", str(e))
            self.is_running = False
            self.is_paused = True
            self.status_label.config(text="Error", foreground="red")
            return

        # Reschedule the next tick
        # The delay_ms controls the *visual* refresh rate.
        self.root.after(delay_ms, self._simulation_tick)

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for controls."""
        self.root.bind('<space>', lambda e: self._toggle_play_pause())
        self.root.bind('r', lambda e: self._reset_simulation())
        self.root.bind('R', lambda e: self._reset_simulation())
        self.root.bind('1', lambda e: self._set_speed(1.0))
        self.root.bind('2', lambda e: self._set_speed(2.0))
        self.root.bind('5', lambda e: self._set_speed(5.0))
        self.root.bind('0', lambda e: self._set_speed(0.5))
        self.root.bind('<Right>', lambda e: self._step_forward())

    # Step button logic
    def _step_forward(self):
        """Advance simulation by one event."""
        if self.step_pause_timer:
            self.root.after_cancel(self.step_pause_timer)
            self.step_pause_timer = None

        # Ensure we are paused
        if not self.is_paused:
            self.is_paused = True
            self.play_button.config(text="▶ Play")
        
        # Check if simulation is over
        next_event_time = self.model.env.peek()
        is_complete = (not self.is_running) or \
                      (self.model.env.now >= self._simulation_time_limit) or \
                      (next_event_time == float('inf'))
                      
        if is_complete:
             self.is_running = False
             self.status_label.config(text="Completed", foreground="green")
             return

        # Set status
        self.status_label.config(text="Stepping...", foreground="blue")
        
        # Run one simulation step
        try:
            # Run until just after the next event
            run_until = min(next_event_time + 0.00001, self._simulation_time_limit)
            self.model.env.run(until=run_until) 
            
            # Schedule a status update back to 'Paused'
            self.step_pause_timer = self.root.after(100, self._auto_pause_after_step)
            
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
            self.is_running = False
            self.status_label.config(text="Error", foreground="red")
    
    # Play/Pause button logic
    def _toggle_play_pause(self):
        """Toggle between play and pause."""
        if self.step_pause_timer:
            self.root.after_cancel(self.step_pause_timer)
            self.step_pause_timer = None

        # If simulation is finished, pressing Play should Reset
        next_event_time = self.model.env.peek()
        is_complete = (not self.is_running) or \
                      (self.model.env.now >= self._simulation_time_limit) or \
                      (next_event_time == float('inf'))

        if is_complete and not self.is_paused: # If already finished, pause it
             self.is_paused = True
        elif is_complete and self.is_paused: # If finished and paused, reset
             self._reset_simulation()
             # After reset, we want to start playing
             self.is_paused = False
             self.is_running = True 
             self.play_button.config(text="⏸ Pause")
             self.status_label.config(text="Running", foreground="green")
             return

        # Standard toggle
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.play_button.config(text="▶ Play")
            self.status_label.config(text="Paused", foreground="orange")
        else:
            self.play_button.config(text="⏸ Pause")
            self.status_label.config(text="Running", foreground="green")
            
            if not self.is_running:
                 # This will be true on the very first play click
                self.is_running = True
    
    def _set_speed(self, multiplier: float):
        """Set simulation speed multiplier."""
        self.speed_multiplier = multiplier
        
        # Adjust animation speed based on multiplier
        # A higher multiplier should make the animation *faster* (smaller delay)
        base_animation_speed = 0.02 # seconds per step
        self.animation_speed = base_animation_speed / (multiplier**0.5) # Use sqrt for less extreme speedup
        
        if multiplier >= 50.0:
            self.animation_speed = 0.0 # Max speed = no animation delay
            
        
        self.speed_label.config(text=f"Current: {multiplier}x")
        
        if multiplier >= 5:
            self.speed_label.config(foreground="red")
        elif multiplier >= 2:
            self.speed_label.config(foreground="orange")
        else:
            self.speed_label.config(foreground="blue")
    
    def _reset_zoom(self):
        """Reset canvas to 1:1 scale using ZoomableCanvas.reset_zoom()."""
        self.canvas.reset_zoom()

    def _fit_view(self):
        """Fit all diagram content into the visible canvas area."""
        self.canvas.fit_view()

    # ------------------------------------------------------------------
    # Zoom-change callback — rescale fonts and entity oval sizes
    # ------------------------------------------------------------------
    # Base font sizes (at scale 1.0)
    _BASE_BLOCK_FONT_DECIDE  = 9   # bold  — DECIDE diamonds
    _BASE_BLOCK_FONT_OTHER   = 9   # normal — all other blocks
    _BASE_ENTITY_FONT        = 8   # bold  — entity number inside ball
    _BASE_ENTITY_RADIUS      = 12  # logical pixels

    def _on_zoom_changed(self, new_scale: float):
        """
        Called by ZoomableCanvas after every zoom (wheel, reset, fit).
        Updates:
          • block-name font sizes  — scaled proportionally
          • entity-number font sizes — scaled proportionally
          • entity oval sizes — rescaled to always be BASE_ENTITY_RADIUS * scale
        """
        # ── Block name fonts ────────────────────────────────────────────
        for name, (shape_id, text_id) in self.block_widgets.items():
            info = self.structure['blocks'].get(name, {})
            if info.get('is_decision'):
                sz   = max(5, int(self._BASE_BLOCK_FONT_DECIDE * new_scale))
                font = ("Arial", sz, "bold")
            else:
                sz   = max(5, int(self._BASE_BLOCK_FONT_OTHER * new_scale))
                font = ("Arial", sz)
            try:
                self.canvas.itemconfig(text_id, font=font)
            except tk.TclError:
                pass

        # ── Entity fonts + oval sizes ───────────────────────────────────
        target_r = max(5, self._BASE_ENTITY_RADIUS * new_scale)
        efont = ("Arial", max(4, int(self._BASE_ENTITY_FONT * new_scale)), "bold")
        for entity_id, (circle_id, text_id) in list(self.entities_on_canvas.items()):
            try:
                # Re-centre the oval at its current centre with the target radius
                coords = self.canvas.coords(circle_id)
                if len(coords) == 4:
                    cx = (coords[0] + coords[2]) / 2
                    cy = (coords[1] + coords[3]) / 2
                    self.canvas.coords(circle_id,
                                       cx - target_r, cy - target_r,
                                       cx + target_r, cy + target_r)
                    self.canvas.coords(text_id, cx, cy - 1)
                self.canvas.itemconfig(text_id, font=efont)
            except tk.TclError:
                pass

    # Reset logic
    def _reset_simulation(self):
        """Stop and reset simulation to initial state."""
        # ── 1. STOP everything immediately ──────────────────────────────
        self.is_running = False
        self.is_paused  = True
        self._animation_generation += 1

        if self.step_pause_timer:
            self.root.after_cancel(self.step_pause_timer)
            self.step_pause_timer = None

        # ── 2. Clear canvas entities ─────────────────────────────────────
        for entity_id, (circle, text) in list(self.entities_on_canvas.items()):
            self.canvas.delete(circle)
            self.canvas.delete(text)
        self.entities_on_canvas.clear()

        # Clear queue/service slots
        self.entity_queue_slots.clear()
        self.entity_service_slots.clear()

        # ── 3. Reset stats ───────────────────────────────────────────────
        self.stats = {
            'total_created': 0,
            'total_disposed': 0,
            'current_wip': 0,
            'simulation_time': 0.0
        }
        for key in list(self.stats_labels.keys()):
            if key.endswith('_util'):
                self.stats_labels[key].config(text="0.00%")
            elif key.endswith('_queue'):
                self.stats_labels[key].config(text="0")
        self._update_stats_display()

        # ── 4. Rebuild model & re-instrument ─────────────────────────────
        self.model      = self.model_builder()
        self.instrument = VisualizationInstrument(self.model, self.event_queue)
        self.structure  = ModelInspector.extract_structure(self.model)

        if hasattr(self, 'custom_positions') and self.custom_positions:
            self.positions = self.custom_positions
        else:
            self.positions = AutoLayout.generate(
                self.structure, self.canvas_width, self.canvas_height
            )

        # ── 5. Redraw (also resets canvas transform to 1:1) ───────────────
        self.canvas.reset_zoom()          # ← reset zoom before redrawing
        self.canvas.delete("all")

        # Clear all canvas-item-ID caches
        self.queue_rect_ids.clear()
        self.service_rect_ids.clear()
        self.connection_line_ids.clear()
        self.connection_paths.clear()
        self.block_widgets.clear()
        self.block_centers.clear()
        self.queue_areas.clear()
        self.service_areas.clear()

        self._draw_blocks()
        self._draw_connections()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # ── 6. Animate layout into place (yEd-style) ─────────────────────
        self._animate_layout_to_positions(self.positions)

        # ── 7. Re-init SimPy generators & update UI ───────────────────────
        self._initialize_simulation()
        self.play_button.config(text="▶ Play")
        self.status_label.config(text="Ready", foreground="green")

    def _animate_layout_to_positions(self, target_positions: dict, frames: int = 30, delay_ms: int = 20):
        """
        Animate blocks smoothly from their current canvas positions to
        target_positions — gives the yEd 'automatic layout animation' feel.
        """
        # Capture current centre of every block widget
        start = {}
        for name, (shape, label_id) in self.block_widgets.items():
            coords = self.canvas.bbox(shape)
            if coords:
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2
                start[name] = (cx, cy)
            else:
                start[name] = target_positions.get(name, (0, 0))

        gen = self._animation_generation  # capture so stale calls exit

        def step(frame):
            if gen != self._animation_generation:
                return  # reset happened; abort
            t = frame / frames
            # Ease-in-out cubic: smooth start and end
            t_eased = t * t * (3 - 2 * t)

            for name, (tx, ty) in target_positions.items():
                if name not in self.block_widgets:
                    continue
                sx, sy = start.get(name, (tx, ty))
                nx = sx + (tx - sx) * t_eased
                ny = sy + (ty - sy) * t_eased
                dx = nx - sx - (nx - sx) * ((frame - 1) / frames if frame > 0 else 0)

                # Compute incremental delta from last frame position
                prev_t = ((frame - 1) / frames) if frame > 0 else 0
                prev_eased = prev_t * prev_t * (3 - 2 * prev_t)
                prev_nx = sx + (tx - sx) * prev_eased
                prev_ny = sy + (ty - sy) * prev_eased
                ddx = nx - prev_nx
                ddy = ny - prev_ny

                shape, label_id = self.block_widgets[name]
                try:
                    self.canvas.move(shape,    ddx, ddy)
                    self.canvas.move(label_id, ddx, ddy)
                except tk.TclError:
                    pass

                # Also move queue / service dashed boxes
                if name in self.queue_areas:
                    qa = self.queue_areas[name]
                    self.queue_areas[name] = (qa[0]+ddx, qa[1]+ddy, qa[2]+ddx, qa[3]+ddy)
                if name in self.service_areas:
                    sa = self.service_areas[name]
                    self.service_areas[name] = (sa[0]+ddx, sa[1]+ddy, sa[2]+ddx, sa[3]+ddy)

                # Update centre cache
                self.block_centers[name] = (tx, ty) if frame == frames else (
                    int(sx + (tx - sx) * t_eased),
                    int(sy + (ty - sy) * t_eased)
                )

            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            if frame < frames:
                self.root.after(delay_ms, lambda: step(frame + 1))

        step(1)


    def _handle_stats_update(self, event):
        """Handle statistics update event."""
        self.stats['simulation_time'] = event.data.get('time', 0)
        
        # Update resource utilization (NOT queue - we calculate that visually)
        for key, value in event.data.items():
            if key.endswith('_util'):
                self.stats[key] = value
        

        
        self._update_stats_display()
    
    def _animate_move_along_path(self, entity_id, circle, text, from_block, to_block, state):
        """
        Animate entity movement.  All target coordinates are read from the
        CURRENT (post-zoom) canvas positions of block shapes and queue rects,
        so entities land correctly at any zoom level.
        """
        # Remove from previous slot
        if from_block:
            if from_block in self.entity_queue_slots and entity_id in self.entity_queue_slots[from_block]:
                self.entity_queue_slots[from_block].remove(entity_id)
                self._reposition_queue(from_block)
            if from_block in self.entity_service_slots and entity_id in self.entity_service_slots[from_block]:
                self.entity_service_slots[from_block].remove(entity_id)
                self._reposition_service(from_block)

        # Case 1: Move from queue to service (within same block)
        if from_block == to_block and state == 'service':
            target_x, target_y = self._current_block_center(to_block)
            self._animate_segment(entity_id, circle, text,
                                  [(target_x, target_y)], 0, to_block, state)
            return

        # Case 2: Move between different blocks — path from CURRENT canvas coords
        path_segments = self._current_connection_path(from_block, to_block)

        if not path_segments:
            # Snap to target
            if state == 'queue' and to_block in self.queue_areas:
                if entity_id not in self.entity_queue_slots.get(to_block, []):
                    self.entity_queue_slots.setdefault(to_block, []).append(entity_id)
                self._reposition_queue(to_block)
                return
            else:
                target_x, target_y = self._current_block_center(to_block)
                try:
                    coords = self.canvas.coords(circle)
                    r = (coords[2] - coords[0]) / 2 if len(coords) == 4 else \
                        max(5, self._BASE_ENTITY_RADIUS * getattr(self.canvas, '_scale', 1.0))
                    self.canvas.coords(circle, target_x - r, target_y - r,
                                               target_x + r, target_y + r)
                    self.canvas.coords(text, target_x, target_y - 1)
                except tk.TclError:
                    pass
                return

        self._animate_segment(entity_id, circle, text, path_segments, 0, to_block, state)

    def _animate_segment(self, entity_id, circle, text, path_segments, index, final_block, final_state, _gen=None):
        """
        Recursively animates one segment of a path.
        FIX: Smooth animation along connector paths.
        Uses _gen to detect and skip animations that were started before the last Reset.
        """
        # Capture generation on first call; abort if stale on subsequent calls
        if _gen is None:
            _gen = self._animation_generation
        if _gen != self._animation_generation:
            return  # reset happened → discard this animation

        if index >= len(path_segments):
            # Animation complete — place entity at CURRENT canvas position
            q_area = self._current_queue_area(final_block)
            s_area = self._current_service_area(final_block)

            if final_state == 'queue' and q_area is not None:
                if entity_id not in self.entity_queue_slots.get(final_block, []):
                    self.entity_queue_slots.setdefault(final_block, []).append(entity_id)
                self._reposition_queue(final_block)
            elif final_state == 'service' and s_area is not None:
                if entity_id not in self.entity_service_slots.get(final_block, []):
                    self.entity_service_slots.setdefault(final_block, []).append(entity_id)
                self._reposition_service(final_block)
            else:
                if entity_id not in self.entity_service_slots.get(final_block, []):
                    self.entity_service_slots.setdefault(final_block, []).append(entity_id)
                self._reposition_service(final_block)
            return

        # Get current position
        try:
            x1, y1, x2, y2 = self.canvas.coords(circle)
            current_x = (x1 + x2) / 2
            current_y = (y1 + y2) / 2
        except:
            return
        
        target_x, target_y = path_segments[index]
        
        dx = target_x - current_x
        dy = target_y - current_y
        
        steps_to_move = max(1, int(self.steps_per_move / len(path_segments)))
        
        # Handle "MAX" speed (no animation)
        if self.speed_multiplier >= 50.0:
            steps_to_move = 1
        
        step_dx = dx / steps_to_move
        step_dy = dy / steps_to_move
        
        def animation_loop(step):
            # Abort if a Reset happened since this animation started
            if _gen != self._animation_generation:
                return
            if step >= steps_to_move:
                # Segment complete, move to next
                self._animate_segment(entity_id, circle, text, path_segments, index + 1, final_block, final_state, _gen)
                return
            
            try:
                self.canvas.move(circle, step_dx, step_dy)
                self.canvas.move(text, step_dx, step_dy)
                # Only call canvas.update() if at max speed
                if self.speed_multiplier >= 50.0:
                    self.canvas.update()
            except tk.TclError:
                return
            
            delay_ms = max(1, int(self.animation_speed * 1000))
            
            # At MAX speed, don't use root.after, just loop
            if self.speed_multiplier >= 50.0:
                animation_loop(step + 1)
            else:
                self.root.after(delay_ms, lambda: animation_loop(step + 1))
        
        animation_loop(0)

    def _reposition_queue(self, block_name):
        """Repositions all entities in a block's queue area using CURRENT canvas coords."""
        q_area = self._current_queue_area(block_name)
        if q_area is None:
            return

        queue = self.entity_queue_slots.get(block_name, [])
        slot_w = max(1, (q_area[2] - q_area[0]) / max(1, len(queue) if queue else 5))
        slot_w = min(slot_w, 24 * max(1.0, getattr(self.canvas, '_scale', 1.0)))
        max_in_row = max(1, int((q_area[2] - q_area[0]) / slot_w))

        for i, entity_id in enumerate(queue):
            if entity_id not in self.entities_on_canvas:
                continue
            circle, text = self.entities_on_canvas[entity_id]
            cx = q_area[0] + (i % max_in_row) * slot_w + slot_w / 2
            cy = (q_area[1] + q_area[3]) / 2
            try:
                coords = self.canvas.coords(circle)
                r = (coords[2] - coords[0]) / 2 if len(coords) == 4 else \
                    max(5, self._BASE_ENTITY_RADIUS * getattr(self.canvas, '_scale', 1.0))
                self.canvas.coords(circle, cx - r, cy - r, cx + r, cy + r)
                self.canvas.coords(text,   cx, cy - 1)
            except tk.TclError:
                continue
        self._update_stats_display()
    
    def _reposition_service(self, block_name):
        """Repositions all entities in a block's service area using CURRENT canvas coords."""
        s_area       = self._current_service_area(block_name)
        service_list = self.entity_service_slots.get(block_name, [])
        scale        = getattr(self.canvas, '_scale', 1.0)

        def _place(entity_id, cx, cy):
            """Move entity circle+text to centre (cx, cy) using its actual radius."""
            if entity_id not in self.entities_on_canvas:
                return
            circle, text = self.entities_on_canvas[entity_id]
            try:
                coords = self.canvas.coords(circle)
                r = (coords[2] - coords[0]) / 2 if len(coords) == 4 else \
                    max(5, self._BASE_ENTITY_RADIUS * scale)
                self.canvas.coords(circle, cx - r, cy - r, cx + r, cy + r)
                self.canvas.coords(text,   cx, cy - 1)
            except tk.TclError:
                pass

        if s_area is None:
            cx, cy = self._current_block_center(block_name)
            for eid in service_list:
                _place(eid, cx, cy)
            self._update_stats_display()
            return

        slot_w = max(1, (s_area[2] - s_area[0]) /
                    max(1, len(service_list) if service_list else 5))
        slot_w = min(slot_w, 24 * max(1.0, scale))
        max_in_row = max(1, int((s_area[2] - s_area[0]) / slot_w))

        for i, entity_id in enumerate(service_list):
            cx = s_area[0] + (i % max_in_row) * slot_w + slot_w / 2
            cy = (s_area[1] + s_area[3]) / 2
            _place(entity_id, cx, cy)
        self._update_stats_display()
    
    def _update_stats_display(self):
        """
        Update statistics labels.
        FIX: Queue counts now reflect VISUAL queue (not SimPy's internal queue).
        """
        for key, label in self.stats_labels.items():
            if key in self.stats:
                value = self.stats[key]
                
                if key == 'simulation_time':
                    label.config(text=f"{value:.1f}")
                elif key.endswith('_util'):
                    label.config(text=f"{value:.2f}%")
                else:
                    label.config(text=str(int(value)))
            
            # Calculate queue count from VISUAL queues
            elif key.endswith('_queue'):
                res_name = key.replace('_queue', '')
                blocks_for_this_resource = self.resource_to_blocks_map.get(res_name, [])
                
                # Sum lengths of visual queues for all blocks using this resource
                total_queue_count = 0
                for block_name in blocks_for_this_resource:
                    total_queue_count += len(self.entity_queue_slots.get(block_name, []))
                
                label.config(text=str(total_queue_count))

            # ADD debugging (temporary):
            elif key.endswith('_queue'):
                res_name = key.replace('_queue', '')
                blocks_for_this_resource = self.resource_to_blocks_map.get(res_name, [])
                
                total_queue_count = 0
                for block_name in blocks_for_this_resource:
                    queue_len = len(self.entity_queue_slots.get(block_name, []))
                    total_queue_count += queue_len
                    # DEBUG: Print to console
                    if queue_len > 0:
                        # Comment out debug print
                        # print(f"[DEBUG] {res_name} @ {block_name}: {queue_len} in queue")
                        pass
                
                label.config(text=str(total_queue_count))
    
    def _auto_pause_after_step(self):
        """Called by timer to re-pause after a step."""
        self.is_paused = True
        self.play_button.config(text="▶ Play")
        self.status_label.config(text="Paused", foreground="orange")
        self.step_pause_timer = None

    # Run method
    def run(self):
        """Start the visualizer (blocks until window closed)."""

        print("=" * 120)        
        print(f"{'Time':<8} | {'Event':<22}  | {'Entity':<15} | {'Resource':<30} | {'Details':<50}")
        print("-" * 120)
        
        self.setup_gui()
        self._initialize_simulation() # Initialize generators
        self.root.mainloop()


# =============================================================================
# Instrumentation for Simulation Model
# =============================================================================
class VisualizationInstrument:
    """
    Instruments a simulation model to send events to visualizer.
    """
    
    def __init__(self, model, event_queue: EventQueue):
        self.model = model
        self.event_queue = event_queue
        self.entity_counter = 0
        self.entity_locations = {}
        
        self._instrument_blocks()
    
    def _instrument_blocks(self):
        """Wrap block methods to send visualization events."""
        from desk.blocks.create_block import CreateBlock
        from desk.blocks.dispose_block import DisposeBlock
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        for name, block in self.model.blocks.items():
            original_process = block.process_entity
            
            if isinstance(block, CreateBlock):
                original_gen = block._generation_process
                block._generation_process = self._wrap_create_generator(
                    original_gen, block
                )
            elif isinstance(block, DisposeBlock):
                block.process_entity = self._wrap_dispose(original_process, block)
            elif isinstance(block, (ProcessBlock, MultiProcessBlock)):
                # Special handling for ProcessBlocks
                block.process_entity = self._wrap_process_with_resource_check(
                    original_process, block
                )
                original_log_start = block.log_start
                block.log_start = self._wrap_log_start(original_log_start, block.name)
            else:
                block.process_entity = self._wrap_process(original_process, block)

            original_log_complete = block.log_complete
            block.log_complete = self._wrap_log_complete(original_log_complete, block.name)

    # Resource-aware process wrapping
    def _wrap_process_with_resource_check(self, original_func, block):
        """
        Wrap ProcessBlock with resource availability checking.
        
        Logic:
        - If resource has available capacity -> go directly to 'service'
        - If resource is full -> go to 'queue' first, then 'service' when seized
        """
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        def wrapped(entity):
            entity_id = self._get_entity_id(entity)
            from_block, old_state = self.entity_locations.get(entity_id, (None, 'service'))
            
            # CHECK: Determine if entity should queue or go directly to service
            should_queue = False
            
            if isinstance(block, ProcessBlock) and block.resource:
                # Check if resource is at full capacity
                resource = block.resource
                units_needed = getattr(block, 'resource_units', 1)
                available_capacity = resource.capacity - resource.count
                
                should_queue = (available_capacity < units_needed)
                
            elif isinstance(block, MultiProcessBlock):
                # Check if ALL required resources are available
                all_available = True
                for resource, units_needed in block.resource_requirements.items():
                    available_capacity = resource.capacity - resource.count
                    if available_capacity < units_needed:
                        all_available = False
                        break
                
                should_queue = not all_available
            
            # SET STATE: Based on resource availability
            new_state = 'queue' if should_queue else 'service'
            
            self.entity_locations[entity_id] = (block.name, new_state)
            
            # Send movement event
            self.event_queue.put(VisualizationEvent(
                event_type='entity_moved',
                timestamp=self.model.env.now,
                data={
                    'entity_id': entity_id,
                    'from_block': from_block,
                    'to_block': block.name,
                    'state': new_state
                }
            ))
            
            self._send_stats_update()
            
            # Small timeout for GUI update
            yield self.model.env.timeout(0.001)
            yield from original_func(entity)
        
        return wrapped

    def _wrap_create_generator(self, original_gen_func, block):
        """Wrap CreateBlock generator to track entity creation."""
        def new_wrapped_generator():
            for item in original_gen_func():
                if hasattr(block, 'entities_created') and block.entities_created > 1: 
                    entity_num = block.entities_created - 1 
                    entity_id = f"{block.entity_prefix}_{entity_num}" 
                    
                    if entity_id not in self.entity_locations:
                        self.event_queue.put(VisualizationEvent(
                            event_type='entity_created',
                            timestamp=self.model.env.now,
                            data={
                                'entity_id': entity_id,
                                'entity_number': entity_num,
                                'block_name': block.name
                            }
                        ))
                        self.entity_locations[entity_id] = (block.name, 'service')
                        self._send_stats_update()
                
                yield item
        
        return new_wrapped_generator
    
    def _wrap_process(self, original_func, block):
        """Wrap block processing to track movements."""
        def wrapped(entity):
            entity_id = self._get_entity_id(entity)
            
            from_block, old_state = self.entity_locations.get(entity_id, (None, 'service'))
            
            is_process = hasattr(block, 'resource') or hasattr(block, 'resource_requirements')
            new_state = 'queue' if is_process else 'service'

            self.entity_locations[entity_id] = (block.name, new_state)

            self.event_queue.put(VisualizationEvent(
                event_type='entity_moved',
                timestamp=self.model.env.now,
                data={
                    'entity_id': entity_id,
                    'from_block': from_block,
                    'to_block': block.name,
                    'state': new_state
                }
            ))
            
            if new_state == 'queue':
                self._send_stats_update()
            
            # Add a small timeout to allow GUI to update
            # This helps visualization feel more "real-time"
            yield self.model.env.timeout(0.001) 
            yield from original_func(entity)
        
        return wrapped

    def _wrap_log_start(self, original_log_start, block_name):
        """
        Wrap log_start to move entity from queue to service.
        
        This is called when resource is actually SEIZED.
        If entity was in queue, it now moves to service.
        """
        def wrapped(entity, resource_name=None):
            original_log_start(entity, resource_name)
            
            entity_id = self._get_entity_id(entity)
            current_block, current_state = self.entity_locations.get(
                entity_id, (block_name, 'queue')
            )
            
            # ONLY send event if entity was actually in queue
            if current_state == 'queue':
                self.entity_locations[entity_id] = (block_name, 'service')
                
                self.event_queue.put(VisualizationEvent(
                    event_type='entity_moved',
                    timestamp=self.model.env.now,
                    data={
                        'entity_id': entity_id,
                        'from_block': block_name,
                        'to_block': block_name,
                        'state': 'service'
                    }
                ))
                self._send_stats_update()
        
        return wrapped

    def _wrap_log_complete(self, original_log_complete, block_name):
        """Wrap log_complete to mark entity as ready to move."""
        def wrapped(entity, resource_name=None):
            original_log_complete(entity, resource_name)
            
            entity_id = self._get_entity_id(entity)
            self.entity_locations[entity_id] = (block_name, 'complete')
        return wrapped
    
    def _wrap_dispose(self, original_func, block):
        """Wrap DisposeBlock to track disposal."""
        def wrapped(entity):
            entity_id = self._get_entity_id(entity)
            from_block, old_state = self.entity_locations.get(entity_id, (None, 'service'))
            
            self.event_queue.put(VisualizationEvent(
                event_type='entity_moved',
                timestamp=self.model.env.now,
                data={
                    'entity_id': entity_id,
                    'from_block': from_block,
                    'to_block': block.name,
                    'state': 'service'
                }
            ))
            
            # Add a small timeout
            yield self.model.env.timeout(0.001)
            yield from original_func(entity)
            
            self.event_queue.put(VisualizationEvent(
                event_type='entity_disposed',
                timestamp=self.model.env.now,
                data={'entity_id': entity_id, 'block_name': block.name}
            ))
            
            if entity_id in self.entity_locations:
                del self.entity_locations[entity_id]
            self._send_stats_update()
        
        return wrapped
    
    def _get_entity_id(self, entity) -> str:
        # Use the entity.id attribute directly
        return entity.id
    
    def _send_stats_update(self):
        """Send statistics update event."""
        stats_data = {'time': self.model.env.now}
        
        if hasattr(self.model, 'resources'):
            for res_name, resource in self.model.resources.items():
                try:
                    # Get blocks using this resource
                    blocks_using_resource = []
                    for block_name, block in self.model.blocks.items():
                        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
                        if isinstance(block, ProcessBlock) and block.resource == resource:
                            blocks_using_resource.append(block_name)
                        elif isinstance(block, MultiProcessBlock) and resource in block.resource_requirements:
                            blocks_using_resource.append(block_name)
                    
                    # This will be set by the GUI later, just use SimPy's count for now
                    if resource.capacity > 0:
                        utilization = (resource.count / resource.capacity) * 100
                    else:
                        utilization = 0.0
                    stats_data[f"{res_name}_util"] = utilization
                except Exception as e:
                    stats_data[f"{res_name}_util"] = 0.0
        
        self.event_queue.put(VisualizationEvent(
            event_type='stats_update',
            timestamp=self.model.env.now,
            data=stats_data
        ))


# =============================================================================
# Enables Zooming Canvas
# =============================================================================
class ZoomableCanvas(tk.Canvas):
    """
    Canvas subclass with mouse-wheel zoom, middle-button pan,
    Reset Zoom and Fit View helpers.

    Zoom transform is tracked as a single cumulative scale factor
    (_scale) plus the canvas-coordinate origin shift (_ox, _oy).

    on_zoom: optional callable(new_scale) called after every zoom change so
    the host can rescale fonts, entity sizes, etc.
    """

    # Base entity radius (logical px at scale 1.0)
    BASE_ENTITY_RADIUS = 12

    def __init__(self, master, on_zoom=None, **kwargs):
        super().__init__(master, **kwargs)

        self._scale   = 1.0
        self._ox      = 0.0
        self._oy      = 0.0
        self._on_zoom = on_zoom   # callback(new_scale)

        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>",   self._on_wheel)
        self.bind("<Button-5>",   self._on_wheel)

        self.bind("<ButtonPress-2>",     self._pan_start)
        self.bind("<B2-Motion>",         self._pan_move)
        self.bind("<Alt-ButtonPress-1>", self._pan_start)
        self.bind("<Alt-B1-Motion>",     self._pan_move)

        self._pan_last = None

    # ------------------------------------------------------------------
    # Internal helper — apply a zoom factor around a canvas point
    # ------------------------------------------------------------------
    def _apply_scale(self, factor: float, cx: float, cy: float):
        """Scale all items by *factor* around canvas point (cx, cy)."""
        self.scale("all", cx, cy, factor, factor)
        self._scale *= factor
        self._ox = cx + (self._ox - cx) * factor
        self._oy = cy + (self._oy - cy) * factor
        self.configure(scrollregion=self.bbox("all"))
        if self._on_zoom:
            self._on_zoom(self._scale)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def logical_to_canvas(self, lx: float, ly: float):
        return (self._ox + lx * self._scale,
                self._oy + ly * self._scale)

    def canvas_to_logical(self, cx: float, cy: float):
        if self._scale == 0:
            return (cx, cy)
        return ((cx - self._ox) / self._scale,
                (cy - self._oy) / self._scale)

    # ------------------------------------------------------------------
    # Zoom / Pan
    # ------------------------------------------------------------------
    def _on_wheel(self, event):
        factor = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        self._apply_scale(factor, cx, cy)

    def _pan_start(self, event):
        self._pan_last = (self.canvasx(event.x), self.canvasy(event.y))

    def _pan_move(self, event):
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        if self._pan_last:
            dx = cx - self._pan_last[0]
            dy = cy - self._pan_last[1]
            self.move("all", dx, dy)
            self._ox += dx
            self._oy += dy
            self.configure(scrollregion=self.bbox("all"))
        self._pan_last = (cx, cy)

    # ------------------------------------------------------------------
    # Reset Zoom / Fit View
    # ------------------------------------------------------------------
    def reset_zoom(self):
        """Restore 1:1 scale and origin."""
        if abs(self._scale - 1.0) < 1e-6 and abs(self._ox) < 1 and abs(self._oy) < 1:
            return
        inv = 1.0 / self._scale
        self.scale("all", self._ox, self._oy, inv, inv)
        self.move("all", -self._ox, -self._oy)
        self._scale = 1.0
        self._ox    = 0.0
        self._oy    = 0.0
        self.configure(scrollregion=self.bbox("all"))
        if self._on_zoom:
            self._on_zoom(self._scale)

    def fit_view(self, padding: int = 40):
        """Scale + translate so all content fits inside the visible area."""
        bbox = self.bbox("all")
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        item_w = max(1, x2 - x1)
        item_h = max(1, y2 - y1)

        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            w, h = int(self["width"]), int(self["height"])

        s = min((w - 2 * padding) / item_w,
                (h - 2 * padding) / item_h)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self._apply_scale(s, cx, cy)

        # Centre the content
        new_bbox = self.bbox("all")
        if new_bbox:
            nx1, ny1, nx2, ny2 = new_bbox
            shift_x = w / 2 - (nx1 + nx2) / 2
            shift_y = h / 2 - (ny1 + ny2) / 2
            self.move("all", shift_x, shift_y)
            self._ox += shift_x
            self._oy += shift_y
            self.configure(scrollregion=self.bbox("all"))


def run_visualization(model_builder, simulation_time: float = 100):
    """
    Run simulation with visualization.
    
    Args:
        model_builder: Function that returns a new simulation model instance
        simulation_time: Total simulation time to run
    """
    visualizer = SimulationVisualizer(model_builder)
    visualizer._simulation_time_limit = simulation_time
    visualizer.run()

