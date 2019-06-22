/**
 * EntitiesController
 *
 * @description :: Server-side actions for handling incoming requests.
 * @help        :: See https://sailsjs.com/docs/concepts/actions
 */
const { timeStringToNumber, getNextTime, numberToTimeString } = require('../utils/time')
const { isEqual, range, random } = require('lodash')

module.exports = {
  listEntities: async (req, res) => {
    const data = await Entities.find()
    res.json({ data, total: data.length })
  },

  createEntity: async (req, res) => {
    const { type, name, location } = req.body
    if (type !== 'SUBSTATION' && type !== 'PHOTOVOLTAIC') {
      return res.badRequest({ error: 'type should be "SUBSTATION" or "PHOTOVOLTAIC"' })
    }
    if (!name) {
      return res.badRequest({ error: 'name should not be empty' })
    }
    if (location) {
      const { lng, lat } = location
      if (!(-180 <= lng && lng <= 180 && -90 <= lat && lat <= 90)) {
        return res.badRequest({ error: 'location has incorrect lng/lat' })
      }
    }

    try {
      res.json({ data: await Entities.create({ type, name, location, status: 'PENDING' }).fetch() })
    } catch (error) {
      res.forbidden({ error: error.message })
    }
  },

  getEntity: async (req, res) => {
    const data = await Entities.findOne({ id: req.param('id') })
    if (!data) {
      return res.notFound({ error: 'Entity not found.' })
    }
    res.json({ data })
  },

  updateHistory: async (req, res) => {
    const entityId = req.param('id')
    const entity = await Entities.findOne({ id: entityId })
    if (!entity) {
      return res.badRequest({ error: 'Entity not found.' })
    }
    await Entities.updateOne({ id: entityId }).set({ status: 'READY', modelUpdatedAt: Date.now() })

    const data = req.body.data.map(({ time, power, data }) => ({
      power,
      data,
      time: timeStringToNumber(time),
    }))

    if (data.find(({ time }) => (time % 900000 !== 0))) {
      return res.badRequest({ error: 'time should be multiple of 15 minutes' })
    }

    const updatedRecords = await Promise.all(
      data.map(
        async ({ power, data, time }) => {
          const prevRecord = await Records.findOrCreate(
            { time, entityId },
            { time, entityId, power, data },
          )
          return (
            prevRecord.power !== power
            || prevRecord.time !== time
            || !isEqual(prevRecord.data, data)
          )
            ? Records.update({ id: prevRecord.id }).set({ power, data, time }).fetch()
            : prevRecord
        }
      )
    )

    return res.json({
      data: updatedRecords
    })
  },

  listHistory: async (req, res) => {
    const entityId = req.param('id')
    const entity = await Entities.findOne({ id: entityId })
    if (!entity) {
      return res.badRequest({ error: 'Entity not found.' })
    }
    const to = timeStringToNumber(req.query.to || Date.now())
    const from = timeStringToNumber(req.query.from || 0)

    const records = await Records.find({ time: { '>=': from, '<=': to }, entityId })

    res.json({ data: records })
  },

  forecast: async (req, res) => {
    const entityId = req.param('id')
    const entity = await Entities.findOne({ id: entityId })
    if (!entity) {
      return res.badRequest({ error: 'Entity not found.' })
    }

    const forecastType = req.param('forecastType')
    if (forecastType !== 'ULTRASHORT15_16' && forecastType !== 'SHORT15_96') {
      return res.badRequest({ error: 'forecastType should be "ULTRASHORT15_16" or "SHORT15_96"' })
    }

    if (entity.status !== 'READY') {
      return res.forbidden({ error: 'Model is not ready, history data is required for forecasting.' })
    }

    if (forecastType === 'ULTRASHORT15_16') {
      const from = getNextTime(
        timeStringToNumber(req.query.from || Date.now()),
        900000,
      )
      // Mock forecast
      const record = (await Records.find({ where: { entityId }, limit: 1 }))[0]
      return res.json({
        data: range(16).map(i => ({
          time: numberToTimeString(from + i * 900000),
          power: record.power * random(0.8, 1.2),
          power_5: null,
          power_95: null,
        })),
      })
    }
    if (forecastType === 'SHORT15_96') {
      const from = getNextTime(
        timeStringToNumber(req.query.from || Date.now()),
        86400000,
      )
      // Mock forecast
      const record = (await Records.find({ where: { entityId }, limit: 1 }))[0]
      return res.json({
        data: range(16).map(i => ({
          time: numberToTimeString(from + i * 900000),
          power: record.power * random(0.8, 1.2),
          power_5: null,
          power_95: null,
        })),
      })
    }
  },
}
