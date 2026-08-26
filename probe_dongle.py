import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from urh.dev.BackendHandler import BackendHandler
from urh.dev.VirtualDevice import Mode, VirtualDevice

bh = BackendHandler()
dev = VirtualDevice(bh, "RTL-SDR", Mode.receive, freq=433.92e6,
                    sample_rate=1e6, gain=40, resume_on_full_receive_buffer=True, raw_mode=True)
dev.start()
old = 0
got_any = False
t0 = time.time()
while time.time() - t0 < 8:
    data = dev.data
    if data is None:
        time.sleep(0.05); continue
    cur = dev.current_index
    if cur != old:
        got_any = True
        print("DATA! index {} (+{} samples)".format(cur, (cur-old) % len(data)))
        break
    time.sleep(0.1)
if not got_any:
    print("NO DATA after 8 s")
try:
    dev.stop("probe done")
except Exception as e:
    print("stop err:", e)
