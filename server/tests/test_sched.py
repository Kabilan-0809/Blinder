import sys
import time
sys.path.insert(0, './')
from server.scheduler.dynamic_frame_scheduler import DynamicFrameScheduler

def test_cooldowns():
    sched = DynamicFrameScheduler()
    mem = {}
    nav = {}

    # Initial capture - should be TRUE because of force flag
    yolo1 = {'object_types': {'car'}, 'clear_path': True, 'objects': [{'label': 'car'}]}
    res1 = sched.should_capture_frame(scene_state=yolo1, navigation_state=nav, session_memory=mem)
    print("Call 1 (New Car): Capture =", res1["capture"], "Rule =", res1.get("rule"))

    # Immediately after, add motorcycle
    sched._last_llm_time = time.time() - 8 
    yolo2 = {'object_types': {'car', 'motorcycle'}, 'clear_path': True, 'objects': [{}, {}]}
    res2 = sched.should_capture_frame(scene_state=yolo2, navigation_state=nav, session_memory=mem)
    print("Call 2 (New Motorcycle at T+8s): Capture =", res2["capture"], "Rule =", res2.get("rule"))
    # Expected: False (cooldown is 15s)

    # 16s pass
    sched._last_llm_time = time.time() - 16
    res3 = sched.should_capture_frame(scene_state=yolo2, navigation_state=nav, session_memory=mem, force=False)
    print("Call 3 (After 16s): Capture =", res3["capture"], "Rule =", res3.get("rule"))
    # Expected: True (default stability window, rule6 won't trigger if prev_obj_types didn't change, but elapsed > 10s so default triggers)

    # Test high value class filtering
    yolo3 = {'object_types': {'car', 'motorcycle', 'dining table'}, 'clear_path': True, 'objects': [{}, {}, {}]}
    sched._last_llm_time = time.time() - 12
    res4 = sched.should_capture_frame(scene_state=yolo3, navigation_state=nav, session_memory=mem)
    print("Call 4 (New Dining Table at T+12s): Capture =", res4["capture"], "Rule =", res4.get("rule"))
    # Expected: Capture = True because elapsed > 10.0s (default stable window), but rule != "rule6_new_objects"

if __name__ == "__main__":
    test_cooldowns()
