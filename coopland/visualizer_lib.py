import tkinter
import dataclasses
from coopland import game_lib


@dataclasses.dataclass
class Visualizer:
    cell_size_px: int = 20
    sec_per_turn: float = 2.0
    move_animation_sec: float = 0.75

    def run(self, game: game_lib.Game, replay: game_lib.AllAgentReplays):
        tk = tkinter.Tk()
        tk.after(1, lambda: tk.focus_force())
        cell_size = self.cell_size_px
        maze = game.maze

        canvas = tkinter.Canvas(
            tk, width=cell_size * maze.width, height=cell_size * maze.height, bg="white"
        )
        canvas.pack()

        self._draw_maze(canvas, maze, game.exit_position)

        agent_widgets = []

        for i, p in enumerate(game.initial_agent_positions):
            agent_widgets.append(self._draw_agent(canvas, i, p))

        def update_loop(t):
            self._update_agent_widgets(canvas, replay, agent_widgets, t, +1)
            canvas.after(int(self.sec_per_turn * 1000), update_loop, t + 1)

        current_t = 0
        n_game_steps = max(map(len, replay))

        def update_once(delta_t):
            nonlocal current_t
            if 0 <= current_t + delta_t <= n_game_steps:
                self._update_agent_widgets(
                    canvas, replay, agent_widgets, current_t, delta_t
                )
                current_t += delta_t

        tk.bind("<Right>", lambda evt: update_once(+1))
        tk.bind("<Left>", lambda evt: update_once(-1))
        # canvas.after(5000, update_loop, 0)

        tk.mainloop()

    def _update_agent_widgets(
        self, canvas, replay: game_lib.AllAgentReplays, widgets, t, dt
    ):
        if not replay:
            return
        cell_size = self.cell_size_px
        for i, (agent_replay, agent_widgets) in enumerate(zip(replay, widgets)):
            try:
                if dt > 0:
                    move, p1, p2 = agent_replay[t]
                else:
                    move, p2, p1 = agent_replay[t + dt]
            except IndexError:
                if dt < 0 and t + dt == len(agent_replay):
                    move, p1, p2 = agent_replay[t + dt - 1]
                    agent_widgets = self._draw_agent(canvas, i, p2)
                    widgets[i] = agent_widgets
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
