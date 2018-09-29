export function assert (condition : any, message? : string) {
  // from http://stackoverflow.com/questions/15313418/javascript-assert
  if (!condition) {
    message = message || 'Assertion failed';
    if (typeof Error !== 'undefined') {
      throw new Error(message);
    }
    throw message; // Fallback
  }
}
