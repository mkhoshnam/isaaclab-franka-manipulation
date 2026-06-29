#!/usr/bin/env bash
set -e

cd ~/Mohammad
mkdir -p isaaclab_ws
cd isaaclab_ws

python3.11 -m venv env_isaaclab
source env_isaaclab/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r ~/Mohammad/isaaclab_exact_requirements.txt

cat > ~/Mohammad/isaaclab_ws/activate_isaaclab.sh <<'EOS'
#!/usr/bin/env bash
cd ~/Mohammad/isaaclab_ws
source env_isaaclab/bin/activate

unset LD_LIBRARY_PATH
unset ROS_DISTRO
unset RMW_IMPLEMENTATION
unset PYTHONPATH
unset PYTHONHOME
unset __NV_PRIME_RENDER_OFFLOAD

export OMNI_KIT_ACCEPT_EULA=YES
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

echo "Isaac Lab env ready."
EOS

chmod +x ~/Mohammad/isaaclab_ws/activate_isaaclab.sh
echo "Done. Test with: source ~/Mohammad/isaaclab_ws/activate_isaaclab.sh"
