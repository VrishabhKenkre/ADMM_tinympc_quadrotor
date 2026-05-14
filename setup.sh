#!/bin/bash
# setup.sh — One-time project setup
# Clones MuJoCo Menagerie Crazyflie model and patches actuator torque limits
# to realistic values (stock limits are placeholders)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══ Quadrotor MPC Project Setup ═══"

# 1. Clone Menagerie (sparse — only Crazyflie)
if [ ! -d "mujoco_menagerie/bitcraze_crazyflie_2" ]; then
    echo "Cloning MuJoCo Menagerie (Crazyflie only)..."
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/google-deepmind/mujoco_menagerie.git
    cd mujoco_menagerie
    git sparse-checkout set bitcraze_crazyflie_2
    cd ..
    echo "  ✓ Cloned"
else
    echo "  ✓ Menagerie already present"
fi

# 2. Patch cf2.xml with realistic torque ranges
# Stock Menagerie uses gear=1e-5 (arbitrary placeholder)
# Real Crazyflie: τ_rp ≈ 0.0069 N·m, τ_yaw ≈ 0.0036 N·m
# (from arm_length × max_motor_thrust = 0.046m × 0.15N)
echo "Patching actuator torque limits..."
CF2_XML="mujoco_menagerie/bitcraze_crazyflie_2/cf2.xml"

sed -i 's|gear="0 0 0 -0.00001 0 0" site="actuation" name="x_moment"|gear="0 0 0 -0.0069 0 0" site="actuation" name="x_moment"|' "$CF2_XML"
sed -i 's|gear="0 0 0 0 -0.00001 0" site="actuation" name="y_moment"|gear="0 0 0 0 -0.0069 0" site="actuation" name="y_moment"|' "$CF2_XML"
sed -i 's|gear="0 0 0 0 0 -0.00001" site="actuation" name="z_moment"|gear="0 0 0 0 0 -0.0036" site="actuation" name="z_moment"|' "$CF2_XML"

echo "  ✓ Patched (τ_rp=0.0069, τ_yaw=0.0036 N·m)"

# 3. Create conda environment
echo ""
echo "To create the conda environment:"
echo "  conda create -n quad_mpc python=3.11 -y"
echo "  conda activate quad_mpc"
echo "  conda install -c conda-forge mujoco casadi numpy scipy matplotlib osqp -y"
echo ""

# 4. Verify
echo "Running verification..."
python3 src/quad_dynamics.py
echo ""
echo "═══ Setup Complete ═══"
echo "Run: python3 src/verify_system.py"
