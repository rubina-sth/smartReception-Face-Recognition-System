var express = require('express');
var fs = require('fs');
var router = express.Router();
let { PythonShell } = require('python-shell');
var createError = require('http-errors');

/* GET users listing. */
router.get('/', callName);

function callName(req, res, next) {
	console.log('identification');
	let options = {
		// mode: 'text',
		// pythonPath: 'path/to/python',
		// pythonOptions: ['-u'], // get print results in real-time
		scriptPath: 'public/python',
		// args: ['value1', 'value2', 'value3']
	};
	PythonShell.run('recogniser.py', options, function (err, results) {
		if (err) {
			console.log(err);
			return next(createError(400));
		}
		console.log('start');
		console.log('results: ', results);
	});
	res.redirect('/');
}


module.exports = router;
