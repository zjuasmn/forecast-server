module.exports = function forbidden(data) {
  const { res } = this

  res.status(403)
  if (data) {
    return res.json(data)
  } else {
    return res.sendStatus(404)
  }
}
