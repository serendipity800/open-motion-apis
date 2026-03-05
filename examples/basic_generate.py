"""Basic generation example — generate a motion and save to NPZ."""

from motion_api.client import MotionClient

client = MotionClient()   # uses localhost:{8081..8084} by default

result = client.generate(
    model="t2m_gpt",
    prompt="a person waves their right hand and bows",
    motion_length=4.0,
    seed=0,
)

print(f"Model  : {result.model}")
print(f"Prompt : {result.prompt}")
print(f"Shapes : {[m.shape for m in result.motions]}")   # [(T, 22, 3)]

result.save_npz("output.npz")
print("Saved → output.npz")
