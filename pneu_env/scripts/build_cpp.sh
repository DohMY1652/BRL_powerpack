echo "Compile start..."
g++ -shared -o pneumatic_simulator.so \
    -fPIC pneumatic_simulator.cpp \
    -fPIC pneumatic_CT.cpp
echo "Compile done!"