# 1  Stop the cluster forcefully (kills raylet & plasma_store).
ray stop --force          # works even if the driver has exited
                          #                         :contentReference[oaicite:1]{index=1}
# 2  Nuke any Python workers Ray left behind.
pkill -9 -u "$USER" -f "ray::_"
pkill -9 -u "$USER" -f "raylet"
pkill -9 -u "$USER" -f "plasma_store"
# 3  Free the GPU if memory is still locked (needs root):
# sudo nvidia-smi --gpu-reset -i <idx>
