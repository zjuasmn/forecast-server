/**
 * EntitiesController
 *
 * @description :: Server-side actions for handling incoming requests.
 * @help        :: See https://sailsjs.com/docs/concepts/actions
 */
const { timeStringToNumber, getNextTime, numberToTimeString } = require('../utils/time')
const { isEqual, range, random } = require('lodash')
const child_process = require('child_process')
const fs = require('fs')
const _ = require('lodash')

const modelTrain = async entity => {
  const entityId = entity.id
  const power = fs.readFileSync(`./data/${entityId}.txt`, 'utf8')
    .trim()
    .split('\n')
    .map(s => Number(s))
  if (power.length < 800) {
    throw new Error('历史数据过少，无法得到预测结果')
  }
  fs.writeFileSync(`./data/${entityId}_trainInput_U.json`, JSON.stringify({
    time: new Date().toISOString(),
    entity: {
      ...entity,
      forecastScale: 'ultrashort',
      type: entity.type === 'SUBSTATION' ? 'LOAD' : entity.type,
    },
    data: {
      [entityId]: {
        power,
      }
    }
  }))
  fs.writeFileSync(`./data/${entityId}_trainInput_S.json`, JSON.stringify({
    time: new Date().toISOString(),
    entity: {
      ...entity,
      forecastScale: 'short',
      type: entity.type === 'SUBSTATION' ? 'LOAD' : entity.type,
    },
    data: {
      [entityId]: {
        power,
      }
    }
  }))
  await new Promise((resolve, reject) => child_process.exec(
    `python ./algorithm/modelTrain.py ./data/${entityId}_trainInput_S.json ./data/${entityId}_model_S.json`,
    (error, stdout, stderr) => {
      fs.writeFileSync(`./log/${entityId}_trainInput_S_stdout.txt`, stdout)
      fs.writeFileSync(`./log/${entityId}_trainInput_S_stderr.txt`, stderr)
      if (error) {
        return reject(error)
      }
      return resolve()
    }
  ))
  await new Promise((resolve, reject) => child_process.exec(
    `python ./algorithm/modelTrain.py ./data/${entityId}_trainInput_U.json ./data/${entityId}_model_U.json`,
    (error, stdout, stderr) => {
      fs.writeFileSync(`./log/${entityId}_trainInput_U_stdout.txt`, stdout)
      fs.writeFileSync(`./log/${entityId}_trainInput_U_stderr.txt`, stderr)
      if (error) {
        return reject(error)
      }
      return resolve()
    },
  ))
}

const modelForecast = async (entity, forecastScale, power) => {
  const entityId = entity.id
  fs.writeFileSync(`./data/${entityId}_forecastInput.json`, JSON.stringify({
    time: new Date().toISOString(),
    entity: {
      ...entity,
      forecastScale,
      type: entity.type === 'SUBSTATION' ? 'LOAD' : entity.type,
    },
    power,
  }))
  const postfix = forecastScale === 'short' ? 'S' : 'U'
  await new Promise((resolve, reject) => child_process.exec(
    `python ./algorithm/modelForecast.py ./data/${entityId}_forecastInput.json ./data/${entityId}_model_${postfix}.json ./data/${entityId}_forecastOutput.json`,
    (error) => {
      if (error) {
        return reject(error)
      }
      return resolve()
    },
  ))
  return JSON.parse(fs.readFileSync(`./data/${entityId}_forecastOutput.json`, 'utf8'))
}

const clusterAnalysis = async (month, data) => {
  fs.writeFileSync(`./data/clusterInput.json`, JSON.stringify({ month, data }))
  await Promise((resolve, reject) => child_process.exec(
    `python ./algorithm/cluster ./data/clusterInput.json ./data/clusterOutput.json`,
    (error) => {
      if (error) {
        return reject(error)
      }
      return resolve()
    }
  ))
  return JSON.parse(fs.readFileSync(`./data/clusterOutput.json`, 'utf8'))
}

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
    if (!req.body || !req.body.data || !_.isArray(req.body.data)) {
      return res.badRequest({ error: 'body.data should be array' })
    }
    await Entities.updateOne({ id: entityId }).set({ status: 'READY' })

    const data = req.body.data.map(({ time, power, data }) => ({
      power,
      data,
      time: timeStringToNumber(time),
    }))

    if (data.find(({ time }) => (time % 900000 !== 0))) {
      return res.badRequest({ error: 'time should be multiple of 15 minutes' })
    }
    const file = fs.openSync(`./data/${entityId}.txt`, 'a')
    fs.writeSync(file, data.map(({ power }) => `${power}\n`).join(''))
    fs.closeSync(file)

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
    const limit = req.query('limit') || 100
    const entity = await Entities.findOne({ id: entityId })
    if (!entity) {
      return res.badRequest({ error: 'Entity not found.' })
    }
    const to = timeStringToNumber(req.query.to || Date.now())
    const from = timeStringToNumber(req.query.from || 0)

    const records = await Records.find(
      { time: { '>=': from, '<=': to }, entityId },
      { limit, sort: { time: -1 } },
    )

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

    const now = new Date()
    const TRAINING_THRESHOLD = 1 * 60 * 1000
    if (!entity.modelUpdatedAt || now - new Date(entity.modelUpdatedAt) > TRAINING_THRESHOLD) {
      await modelTrain(entity)
      await Entities.updateOne({ id: entityId }).set({ status: 'READY', modelUpdatedAt: Date.now() })
    }
    const from = forecastType === 'ULTRASHORT15_16'
      ? getNextTime(
        timeStringToNumber(req.query.from || Date.now()),
        900000,
      )
      : getNextTime(
        timeStringToNumber(req.query.from || Date.now()),
        86400000,
        16 * 3600000,
      )
    const forecastScale = forecastType === 'ULTRASHORT15_16'
      ? 'ultrashort'
      : 'short'
    const forecastLength = forecastType === 'ULTRASHORT15_16'
      ? 16
      : 96
    const records = (await Records.find({
      where: { entityId, time: { '>=': from - 7 * 86400 * 1000, '<': from } },
      sort: { time: 1 }
    }))
    const recordByTime = _.keyBy(records, ({ time }) => time)
    const forecastOutput = await modelForecast(
      entity,
      forecastScale,
      _.range(from - 7 * 86400 * 1000, from, 900000)
        .map(time => (recordByTime[time] || { power: 0 }).power),
    )
    return res.json({
      data: range(forecastLength).map(i => ({
        time: numberToTimeString(from + i * 900000),
        power: forecastOutput[i]['0.5'],
        power_5: forecastOutput[i]['0.05'],
        power_10: forecastOutput[i]['0.1'],
        power_25: forecastOutput[i]['0.25'],
        power_75: forecastOutput[i]['0.75'],
        power_90: forecastOutput[i]['0.9'],
        power_95: forecastOutput[i]['0.95'],
      })),
    })
  },

  cluster: async (req, res) => {
    const month = req.query('month') || `${new Date().getFullYear()}-${new Date().getMonth() + 1}`
    const ids = _.union(req.body.ids)
    if (!ids.length) {
      return res.badRequest({ error: 'ids should not be empty' })
    }
    const entities = await Entities.find({ id: { $in: ids } })
    if (entities.length !== ids.length) {
      return res.badRequest({ error: `Entity not found: found ${entities.map(({ id }) => id).join(',')}` })
    }

    const startTime = new Date(month).getTime()
    const endTime = new Date(month).setMonth(new Date(month).getMonth() + 1).getTime()
    const records = await Records.find({ entityId: { $in: ids }, time: { $gte: startTime, $lt: endTime } })

    const clusterResult = await clusterAnalysis(
      month,
      entities.map(({ id, name }) => ({
        id,
        name,
        power: (() => {
          const recordByTime = _.keyBy(
            records.find(({ entityId }) => entityId === id),
            ({ time }) => time
          )
          return _.range(startTime, endTime, 15 * 60 * 1000)
            .map(t => recordByTime[t] ? recordByTime[t].power : 0)
        })(),
      })),
    )
    return res.json({
      data: clusterResult,
    })
  },
}
