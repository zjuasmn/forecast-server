module.exports = {
  timeStringToNumber: (s) => new Date(s).getTime(),

  numberToTimeString: (n) => new Date(n).toISOString(),

  getNextTime: (t, span) => Math.ceil(t / span) * span,
}
