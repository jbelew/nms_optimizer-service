CXX = em++
CXXFLAGS = -std=c++17 -O3 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORTED_RUNTIME_METHODS=ccall,cwrap -s INVOKE_RUN=0 -s "EXPORT_NAME='createModule'" -s NO_EXIT_RUNTIME=1 -lembind
LDFLAGS = -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 -s "EXPORT_NAME='createModule'" -s NO_EXIT_RUNTIME=1 -lembind
SRCDIR = wasm/src
OBJDIR = wasm/obj
TARGET_JS = static/js/solver.js
TARGET_WASM = static/js/solver.wasm
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))

all: $(TARGET_JS)

$(TARGET_JS): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: all
	python3 wasm/test/generate_golden.py
	node --experimental-modules wasm/test/test_wasm.mjs

clean:
	rm -rf $(OBJDIR) $(TARGET_JS) $(TARGET_WASM) wasm/test/golden.json

.PHONY: all clean test