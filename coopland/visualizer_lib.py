import tkinter
import dataclasses


@dataclasses.dataclass
class Visualizer:
    cell_size_px: int = 20

    def run(self, maze):
        tk = tkinter.Tk()
        cell_size = self.cell_size_px

        canvas = tkinter.Canvas(
            tk, width=cell_size * maze.width, height=cell_size * maze.height, bg="white"
        )
        canvas.pack()

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

        tk.mainloop()
