import tkinter
import dataclasses
from coopland import game_lib


@dataclasses.dataclass
class Visualizer:
    cell_size_px: int = 20
    sec_per_turn: float = 1.5
    move_animation_sec: float = 0.5
    autoplay: bool = False
    title: str = "coopland"
    _position = None

    def run(self, game: game_lib.Game, replay: game_lib.AllAgentReplays):
        tk = tkinter.Tk()
        tk.title(self.title)
        if self._position is not None:
            tk.geometry(f"+{self._position[0]}+{self._position[1]}")
        tk.after(1, lambda: tk.focus_force())
        cell_size = self.cell_size_px
        maze = game.maze

        top_info_frame = tkinter.Frame(tk)
        top_info_frame.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        info_label = tkinter.Label(
            top_info_frame,
            text=f"world={maze.width}x{maze.height} seed={maze.generation_seed}\n"
            f"agent={game.n_agents} x {game.get_agent_name()}",
            justify=tkinter.LEFT,
        )
        info_label.pack(side=tkinter.LEFT)

        canvas = tkinter.Canvas(
            tk,
            width=cell_size * maze.width,
            height=cell_size * maze.height,
            bg="white",
            bd=1,
            relief="ridge",
        )
        canvas.pack()

        bottom_status_frame = tkinter.Frame(tk)
        bottom_status_frame.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
        status_label = tkinter.Label(bottom_status_frame)
        status_label.pack(side=tkinter.LEFT)
        debug_label = tkinter.Label(bottom_status_frame, justify=tkinter.RIGHT)
        debug_label.pack(side=tkinter.RIGHT)

        self._draw_maze(canvas, maze, game.exit_position)

        agent_widgets = []

        for i, p in enumerate(game.initial_agent_positions):
            agent_widgets.append(self._draw_agent(canvas, i, p))

        def update_loop(t):
            moves_with_i = self._update_agent_widgets(
                canvas, replay, agent_widgets, t, +1
            )
            update_debug_label(moves_with_i)
            canvas.after(int(self.sec_per_turn * 1000), update_loop, t + 1)

        current_t = 0
        n_game_steps = max(map(len, replay))
        current_replay_loop_token = 0
        replay_loop_runs = False

        def replay_one_step(delta_t):
            nonlocal current_t, replay_loop_runs, current_replay_loop_token
            if 0 <= current_t + delta_t <= n_game_steps:
                moves_with_i = self._update_agent_widgets(
                    canvas, replay, agent_widgets, current_t, delta_t
                )
                update_debug_label(moves_with_i)
                current_t += delta_t
            if current_t + delta_t > n_game_steps:
                replay_loop_runs = False
                current_replay_loop_token += 1
                if self.autoplay:
                    self._position = tk.winfo_x(), tk.winfo_y()
                    tk.after(1000, tk.destroy)
            update_status_label()

        def toggle_replay_loop():
            nonlocal current_replay_loop_token, replay_loop_runs
            if replay_loop_runs:
                current_replay_loop_token += 1
                replay_loop_runs = False
                update_status_label()
            else:

                def loop(token):
                    if token == current_replay_loop_token:
                        replay_one_step(+1)
                        if replay_loop_runs:
                            canvas.after(int(1000 * self.sec_per_turn), loop, token)

                replay_loop_runs = True
                loop(current_replay_loop_token)

        def update_status_label():
            status_label.config(
                text=f"{'>' if replay_loop_runs else '||'} {current_t} / {n_game_steps}"
            )

        def update_debug_label(moves_with_i):
            texts = [
                f"{i+1}: {move.debug_text}"
                for i, move in moves_with_i
                if getattr(move, "debug_text", None)
            ]
            if texts:
                debug_label.config(text="\n".join(texts))
            else:
                debug_label.config(text="")

        update_status_label()
        tk.bind("<Right>", lambda evt: replay_one_step(+1))
        tk.bind("<Left>", lambda evt: replay_one_step(-1))
        tk.bind("<space>", lambda evt: toggle_replay_loop())

        if self.autoplay:
            toggle_replay_loop()

        tk.mainloop()

    def _update_agent_widgets(
        self, canvas, replay: game_lib.AllAgentReplays, widgets, t, dt
    ):
        if not replay:
            return
        cell_size = self.cell_size_px
        moves_done = []
        for i, (agent_replay, agent_widgets) in enumerate(zip(replay, widgets)):
            try:
                if dt > 0:
                    move, p1, p2 = agent_replay[t]
                else:
                    move, p2, p1 = agent_replay[t + dt]
                moves_done.append((i, move))
            except IndexError:
                if dt < 0 and t + dt == len(agent_replay):
                    move, p1, p2 = agent_replay[t + dt - 1]
                    agent_widgets = self._draw_agent(canvas, i, p2)
                    widgets[i] = agent_widgets
                    moves_done.append((i, move))
                else:
                    for w in agent_widgets:
                        canvas.delete(w)
                    widgets[i] = ()
            else:
                assert agent_widgets
                if p1 != p2:
                    dx = cell_size * (p2[0] - p1[0])
                    dy = cell_size * (p2[1] - p1[1])
                    self._animated_move(canvas, agent_widgets, dx, dy)
        return moves_done

    def _animated_move(self, canvas, widgets, dx, dy):
        anim_timestep_ms = 1000 // 25
        anim_time_ms = int(self.move_animation_sec * 1000)
        n_steps = anim_time_ms // anim_timestep_ms
        dx_dt = dx // n_steps
        dy_dt = dy // n_steps
        last_dx = dx - dx_dt * (n_steps - 1)
        last_dy = dy - dy_dt * (n_steps - 1)

        def anim_fn(steps_left):
            steps_left -= 1
            if steps_left > 0:
                for w in widgets:
                    canvas.move(w, dx_dt, dy_dt)
                canvas.after(anim_timestep_ms, anim_fn, steps_left)
            else:
                for w in widgets:
                    canvas.move(w, last_dx, last_dy)

        anim_fn(n_steps)

    def _draw_agent(self, canvas, i, p):
        cell_size = self.cell_size_px
        padding = cell_size // 5
        half_cell_size = cell_size // 2
        x, y = p
        w1 = canvas.create_oval(
            cell_size * x + padding,
            cell_size * y + padding,
            cell_size * (x + 1) - padding,
            cell_size * (y + 1) - padding,
            fill="#3e0",
        )
        w2 = canvas.create_text(
            cell_size * x + half_cell_size,
            cell_size * y + half_cell_size,
            text=str(i + 1),
        )
        return w1, w2

    def _draw_maze(self, canvas, maze, exit_position):
        cell_size = self.cell_size_px

        for x in range(maze.width):
            for y in range(maze.height):
                if x > 0:
                    if not maze.has_path(x, y, "west"):
                        canvas.create_line(
                            cell_size * x,
                            cell_size * y,
                            cell_size * x,
                            cell_size * (y + 1),
                            width=1,
                        )
                if y > 0:
                    if not maze.has_path(x, y, "north"):
                        canvas.create_line(
                            cell_size * x,
                            cell_size * y,
                            cell_size * (x + 1),
                            cell_size * y,
                            width=1,
                        )

        padding = cell_size // 7
        e_x, e_y = exit_position
        canvas.create_rectangle(
            cell_size * e_x + padding,
            cell_size * e_y + padding,
            cell_size * (e_x + 1) - padding,
            cell_size * (e_y + 1) - padding,
            outline="#d40",
            fill=None,
            width=4,
        )
