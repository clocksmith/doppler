const s = "const s = $;\nconsole.log(s.replace('$', JSON.stringify(s)));";
console.log(s.replace('$', JSON.stringify(s)));
