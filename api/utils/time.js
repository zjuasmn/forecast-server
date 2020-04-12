module.exports = {
  timeStringToNumber: (s) => new Date(s).getTime(),

  numberToTimeString: (n) => new Date(n).toISOString(),

  getNextTime: (t, span, offset = 0) => Math.ceil((t - offset) / span) * span + offset,
}
