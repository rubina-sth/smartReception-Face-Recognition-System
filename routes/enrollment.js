var express = require('express');
var fs = require('fs');
var router = express.Router();
let { PythonShell } = require('python-shell');
var createError = require('http-errors');

router.get('/', function (req, res, next) {
	res.render('enrollment');
});

router.post('/success', function (req, res, next) {
	console.log('post 1');
	var dir = `./public/images/${req.body.folder}`;
	if (!fs.existsSync(dir)) {
		fs.mkdirSync(dir);
	}
	var img = [req.body.image0, req.body.image1, req.body.image2, req.body.image3,req.body.image4];
	var s = [1, 2, 3, 4, 5];
	img.forEach(function (item, index) {
		var data = item.replace(/^data:image\/\w+;base64,/, "");
		var buf = new Buffer(data, 'base64');
		fs.writeFile(`./${dir}/${s[index]}.png`, buf, function (err, done) {
			if (err) {
				console.log(error);
				return next(createError(400));
			}
		});
	})
	next();

}, callName);

// router.get('/run', callName);

function callName(req, res, next) {
	console.log('post 2');
	let options = {
		// mode: 'text',
		// pythonPath: 'path/to/python',
		// pythonOptions: ['-u'], // get print results in real-time
		scriptPath: 'public/python',
		// args: ['value1', 'value2', 'value3']
	};
	PythonShell.run('test.py', options, function (err, results) {
		if (err) {
			console.log(err);
			return next(createError(400));
		}
		console.log('start');
		console.log('results: %j', results);
		res.render('success');
	});
}

module.exports = router;
