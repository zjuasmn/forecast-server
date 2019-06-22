module.exports = function notFound(data) {
  const { res } = this

  res.status(404)
  if (data) {
    return res.json(data)
  } else {
    return res.sendStatus(404)
  }
}
