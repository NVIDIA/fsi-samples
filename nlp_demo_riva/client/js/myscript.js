function clicked() {
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            console.log(this.responseText);
            var jsonResponse = JSON.parse(this.responseText)[0];
		//
            	if (jsonResponse === ''){
            	    document.getElementById("context").innerHTML = 'Sorry, I do not know the answer. Please fine tune the BERT network';
            	}
            	else{
            	    document.getElementById("context").innerHTML = 'Answer: '+ jsonResponse;
            	}

	    function extra(){
            	if (jsonResponse === ''){
            	    playTheText('Sorry, I do not know the answer. Please fine tune the BERT network');
            	}
            	else{
            	    playTheText('The answer is '+jsonResponse);
            	}
	    }

        function say_question(){
            playTheText(question_doc, extra);
        }
            say_question();
            // playTheText('Question', say_question);
        }
    };

    xhttp.open("POST", "infer", true);
    xhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    var para_doc = document.getElementById("para").value;
    var question_doc = document.getElementById("question").value;
    xhttp.send(JSON.stringify({ "para": para_doc, "question": question_doc }));
};


var Sound = (function () {
        var df = document.createDocumentFragment();
        return function Sound(src, callback) {
                    var snd = new Audio(src);
                    df.appendChild(snd); // keep in fragment until finished playing
                    snd.addEventListener('ended', function () {df.removeChild(snd);
			    if (callback != null){
				    callback();
			    }
		    });
                    snd.play();
                    return snd;
                }
}());
// then do it
// var snd = Sound("data:audio/wav;base64," + base64string);
//
function playTheText(inputs, callback){
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            console.log(this);
            var response = this.response;
            // then do it
             var snd = Sound("data:audio/wav;base64," + response, callback);
        }
    };

    xhttp.open("POST", "tacotron", true);
    xhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhttp.send(JSON.stringify({ "text": inputs}));
}

function playIt() {
    var inputs = document.getElementById("question").value;
    playTheText(inputs);
};

function selected(){
    document.getElementById("question").value = document.getElementById("examples").value;
};

function newOption(text){
    opt = document.createElement('option');
    opt.value = text;
    opt.innerText = text;
    return opt;
};

function getText(){
    // read text from URL location
    var request = new XMLHttpRequest();
    request.open('GET', 'doc', true);
    request.send(null);
    request.onreadystatechange = function () {
        if (request.readyState === 4 && request.status === 200) {
            var type = request.getResponseHeader('Content-Type');
            if (type.indexOf("text") !== 1) {
			    document.getElementById("para").value = request.responseText;
                //return request.responseText;
            }
        }
    }

    var request2 = new XMLHttpRequest();
    request2.open('GET', 'questions', true);
    request2.send(null);
    request2.onreadystatechange = function () {
        if (request2.readyState === 4 && request2.status === 200) {
            var type = request2.getResponseHeader('Content-Type');
            if (type.indexOf("text") !== 1) {
	            var jsonResponse = JSON.parse(this.responseText);
		    var question;
      	            var selection = document.getElementById("examples");
		    for (var id=0; id <jsonResponse.length; id++) {
	               var optionEle = newOption(jsonResponse[id].trim());
	               selection.add(optionEle);
		    }
            }
        }
    }

};

window.addEventListener('load', (event) => {
	getText();
	//console.log('loaded');
});
