echo "[ INFO] (Pneumatic simulator C++) Build ==> Start!"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")/../src/pneu_env/"

mkdir -p "${SCRIPT_DIR}/cpp/build"
mkdir -p "${SCRIPT_DIR}/cpp/lib"
cd "${SCRIPT_DIR}/cpp/build"

g++ -c -fPIC "${SCRIPT_DIR}/cpp/src/PneumaticSystem.cpp" -I"${SCRIPT_DIR}/cpp/include"
g++ -c -fPIC "${SCRIPT_DIR}/cpp/src/PneumaticLogger.cpp" -I"${SCRIPT_DIR}/cpp/include"
g++ -c -fPIC "${SCRIPT_DIR}/cpp/src/PneumaticSimulator.cpp" -I"${SCRIPT_DIR}/cpp/include"
g++ -c -fPIC "${SCRIPT_DIR}/cpp/src/extern_functions.cpp" -I"${SCRIPT_DIR}/cpp/include"

g++ -shared -o "${SCRIPT_DIR}/cpp/lib/libPneumaticSimulator.so" \
    ${SCRIPT_DIR}/cpp/build/PneumaticSystem.o \
    ${SCRIPT_DIR}/cpp/build/PneumaticLogger.o \
    ${SCRIPT_DIR}/cpp/build/PneumaticSimulator.o \
    ${SCRIPT_DIR}/cpp/build/extern_functions.o

# g++ -o "${SCRIPT_DIR}/main_executable" "${SCRIPT_DIR}/main.cpp" -I"${SCRIPT_DIR}/include" -L"${SCRIPT_DIR}/lib" -lPneumaticSimulator

if [ $? -eq 0 ]; then
    echo "[ INFO] (Pneumatic simulator C++) Build ==> Success!"
else
    echo "[ERROR] (Pneumatic simulator C++) Build ==> Fail!"
    exit 1
fi
