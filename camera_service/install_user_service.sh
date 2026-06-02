#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="woosh-camera-service.service"
REPO_DIR="/home/wooshrobot/bai/hand_eye_calibration"
UNIT_SRC="${REPO_DIR}/camera_service/systemd/${SERVICE_NAME}"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DIR}/${SERVICE_NAME}"

mkdir -p "${UNIT_DIR}"
cp "${UNIT_SRC}" "${UNIT_DST}"
systemctl --user daemon-reload
systemctl --user enable "${SERVICE_NAME}"
systemctl --user restart "${SERVICE_NAME}"

echo "[OK] installed: ${UNIT_DST}"
echo "[OK] status: systemctl --user status ${SERVICE_NAME}"
echo "[NOTE] For boot without login, run once with sudo:"
echo "      sudo loginctl enable-linger ${USER}"
