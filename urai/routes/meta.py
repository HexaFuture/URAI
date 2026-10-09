"""The console page and the machine-readable quick start at ``GET /api``."""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

STATIC = Path(__file__).resolve().parent.parent/'static'

QUICK_START = {
    'service': 'URAI · Universal Robot–Agent Interface',
    'quick_start': [
        {'step': 1, 'call': 'POST /api/observe', 'body': {'preserve_draft': False},
         'gives': 'observation_id. GET /api/observation.jpg returns the observed frame as JPEG bytes; the pixels of '
                  'every stroke refer to this frame. The "image" field of the JSON reply is a data URL for browsers: '
                  'strip the "data:image/jpeg;base64," prefix before decoding it.'},
        {'step': 2, 'call': 'POST /api/skills/<name>',
         'body': {'observation_id': 'from step 1', 'arm': 'left', 'pixels': [[396, 226], [560, 400]], 'commit': True},
         'gives': 'A draft for one skill, saved when commit is true, with its revision. GET /api/skills lists every '
                  'skill with the stroke it expects, its arguments (ranges and defaults) and its limits. Drafting '
                  'never moves the robot.'},
        {'step': 3, 'call': 'POST /api/preview', 'body': {'expected_revision': 'from step 2'},
         'gives': 'preview_id. Full IK, joint-limit, table, self-collision and inter-arm checks of the approach and the '
                  'task; a refusal says which check failed and by how much.'},
        {'step': 4, 'call': 'POST /api/execute', 'body': {'preview_id': 'from step 3'},
         'gives': 'Starts the motion. Poll GET /api/state until execution.state is completed, error or cancelled.'},
    ],
    'finish': 'POST /api/home {"arms": ["left", "right"]} returns the arms to the all-zero joint pose.',
    'two_arms': 'Call POST /api/skills/<name> once per arm without commit and submit both drafts together with '
                'PUT /api/draft {"observation_id": ..., "arms": {"left": ..., "right": ...}}, then steps 3 and 4. '
                'Both task segments start together after both arms reach their start poses. A second PUT with '
                '{"merge": true} keeps the arm it does not name (same observation_id only).',
    'toss': 'Tossing is not a /api/skills/<name> call: POST /api/tasks queues pick-and-toss items, each planned, '
            'checked and executed in turn. Its fields are listed under the "toss" entry of GET /api/skills.',
    'motion': {
        'moves_the_robot': ['POST /api/execute', 'POST /api/home', 'POST /api/tasks'],
        'may_send_robot_commands': {
            'POST /api/preview': 'With auto_prepare.restore_can on (default), an arm in drag-teach mode or disabled by a '
                                 'protection stop is taken back under CAN control first: the service waits for both '
                                 'arms to stand still for 1 s, then exits teach mode / re-enables the motors and holds '
                                 'the measured pose. This sends CAN commands but commands no motion away from the '
                                 'current pose.',
            'POST /api/execute': 'A preview older than 120 s is regenerated from the same draft (with a fresh '
                                 'observation when auto_prepare.refresh_observation is on) and then executed.',
        },
        'after_execute': 'When a drafted motion completes, the arms that moved return home automatically '
                         '(except the bottle-cap stages that must keep holding the bottle).',
        'stop': 'POST /api/cancel stops the motion and holds position. It is a software stop; keep the hardware '
                'emergency stop within reach.',
    },
    'notes': [
        'Pixels are image coordinates with the origin at the top-left; drawing on an object is enough, the depth '
        'image locates it. Flat cloth that cannot be segmented is pinched where the stroke starts when pinch is set.',
        'Drafting endpoints refuse an observation_id that is not the latest observation.',
        'Observe again after the robot has moved; with auto_prepare.refresh_observation on, preview refreshes the '
        'observation itself and keeps the drafted world-frame path.',
        'Return the arms home before observing: an arm hovering over a dark object merges with it in the depth image.',
    ],
    'other_endpoints': {
        'GET /api/skills': 'skill catalogue: stroke kind, arguments, limits',
        'GET /api/state': 'robot state, execution progress, current observation and draft revision',
        'GET /api/settings, PUT /api/settings': 'pose tolerance, motion profile, table checks, tracking limit, '
                                                'automatic preparation',
        'POST /api/objects': 'raised objects in this frame that the depth image can see, with graspability',
        'POST /api/cancel': 'stop and hold',
    },
    'docs': ['README.md', 'docs/api.md', 'docs/safety.md'],
}


def build_router():
    router = APIRouter()

    @router.get('/')
    def index():
        return FileResponse(STATIC/'index.html')

    @router.get('/favicon.ico', include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @router.get('/api')
    def api_index():
        """The shortest path from a stroke to a motion, for a caller without a browser."""
        return QUICK_START

    return router
