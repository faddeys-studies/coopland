import copy
import threading
import json
import requests
import argparse
import time
from typing import cast
from coopland.maze_lib import Direction
from coopland.game_lib import Game, AllAgentReplays
from tornado import web, ioloop
from coopland.visualizer_lib.impl import Visualizer


class VisualizerServer:
    def __init__(self, port):
        self._lock = threading.Lock()
        self._port = port
        self._current_replay = None
        self._thread = threading.Thread(target=self._server_thread, daemon=True)
        self._thread.start()

    def add_replay(self, game_id, game, replays):
        with self._lock:
            self._current_replay = game_id, game, replays

    def _server_thread(self):
        def get_replay():
            with self._lock:
                return self._current_replay

        loop = ioloop.IOLoop()
        loop.make_current()

        app = web.Application(
            [(r"/get-replay", _RequestHandler, {"get_replay": get_replay})]
        )
        app.listen(self._port)
        print("start replays server at port ", self._port)
        loop.start()


class _RequestHandler(web.RequestHandler):
    def initialize(self, get_replay=None) -> None:
        self._get_replay = get_replay

    def get(self):
        replay = self._get_replay()
        if replay is None:
            raise web.HTTPError(404)
        message = _serialize_game_replay(*replay)
        self.set_header("Content-Type", "application/json")
        self.write(message)


def _serialize_game_replay(game_id, game, replays):
    game_copy = copy.copy(game)
    game_copy.agent_fn = None
    replays_copy = [
        [
            (str(move.direction.value), from_pos, to_pos)
            for move, from_pos, to_pos in replay
        ]
        for replay in replays
    ]
    message = json.dumps([game_id, game_copy.serialize(), replays_copy])
    return message


def _load_game_replay(message):
    game_id, game_str, replays = json.loads(message)
    game = Game.from_serialized(game_str)
    replays = [
        [(DummyMove(move), from_pos, to_pos) for move, from_pos, to_pos in replay]
        for replay in replays
    ]
    return game_id, game, cast(AllAgentReplays, replays)


class DummyMove:
    def __init__(self, d):
        self.direction = Direction(d)


def download_replay(address):
    url = f"http://{address}/get-replay"
    response = requests.get(url)
    if response.status_code == 404:
        return None
    data = response.content.decode("utf-8")
    return _load_game_replay(data)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("server_address")
    server_address = cli.parse_args().server_address

    visualizer = Visualizer(
        cell_size_px=50, sec_per_turn=0.5, move_animation_sec=0.4, autoplay=True
    )
    while True:
        item = download_replay(server_address)
        if item is None:
            time.sleep(10)
            continue
        game_id, game, replays = item
        visualizer.title = f"game #{game_id}"
        visualizer.run(game, replays)


if __name__ == "__main__":
    main()
