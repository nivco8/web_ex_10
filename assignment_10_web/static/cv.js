function print_my_name() {
	console.log("Niv Cohen")
}


var phone_template = /^[0]{1}[0-9]{9}$/;
var email_template = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;

function check_validiation(page){
	var phone_num = page.phone_number
	var email_add = page.email
	var count = 0
	if (!phone_template.test(phone_num)) {
        alert("WARNING: Phone number is not valid");
		count++;
    }
    if (!email_template.test(email_add)) {
        alert("WARNING: Email address is not valid");
		count++;
	}
	if (count>0){
		return false;
	}
	else{
		return true;
	}
}
