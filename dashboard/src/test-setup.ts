// Global test setup — stubs browser APIs missing from jsdom

// Recharts' ResponsiveContainer uses ResizeObserver; jsdom does not provide it.
// A no-op class is sufficient since chart rendering is not asserted in tests.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Only assign if not already defined (avoids clobbering real implementations)
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
}
