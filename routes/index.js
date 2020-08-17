var express = require('express');
var fs = require('fs');
var router = express.Router();
let { PythonShell } = require('python-shell');
var createError = require('http-errors');

/* GET home page. */
router.get('/', function (req, res, next) {
    res.render('index');
});


module.exports = router;
