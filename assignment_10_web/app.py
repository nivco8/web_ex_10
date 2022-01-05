from flask import Flask, redirect, url_for, render_template, request, session
from pages.assignment10.assignment10 import assignment10


app = Flask(__name__)
app.secret_key = "123"


def set_users():
    users_list = [{'first name': 'Yossi', 'last name': 'Cohen', 'email': 'yo@gmail.com'},
                  {'first name': 'Aharon', 'last name': 'Aharoni', 'email': 'ah@gmail.com'},
                  {'first name': 'Oren', 'last name': 'Hazanov', 'email': 'or@gmail.com'},
                  {'first name': 'Ahlan', 'last name': 'Dvori', 'email': 'ahdv@gmail.com'},
                  {'first name': 'Amihai', 'last name': 'Yossi', 'email': 'ah@gmail.com'}]
    return users_list


@app.route('/home_page', methods=['GET', 'POST'])
@app.route('/home')
@app.route('/')
def home_page():
    return render_template('cv.html')


@app.route('/assignment8', methods=['GET', 'POST'])
def details():
    return render_template('assignment8.html',
                           degrees='bsc',
                           hobbies=('playing soccer', 'playing guitar', 'swimming'),
                           visited_places=['costa rica', 'brazil', 'Peru', 'argentina', 'amsterdam'],
                           like_the_course=True)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return render_template('assignment9.html')


@app.route('/assignment9', methods=['GET', 'POST'])
def forms():
    if request.method == 'GET':
        if 'search_key' in request.args:
            search_key = request.args['search_key']
            users_list = set_users()
            matches = []
            if search_key:
                for user in users_list:
                    if user["first name"] == search_key or user["last name"] == search_key or user[
                        "email"] == search_key:
                        matches.append(user)
            else:
                matches = users_list
            if session.get('login'):
                return render_template('assignment9.html', matches_list=matches, nick_name=session.get('nick'))
            else:
                return render_template('assignment9.html', matches_list=matches)
        else:
            return render_template('assignment9.html')
    if request.method == 'POST':
        first = request.form['first']
        last = request.form['last']
        nickname = request.form['nickname']
        email = request.form['email']
        session['nickname'] = nickname
        session['login'] = True
    if session.get('login'):
        return render_template('assignment9.html', nick_name=session.get('nickname'))
    return render_template('assignment9.html')


# assignment10
app.register_blueprint(assignment10)


if __name__ == '__main__':
    app.run(debug=True)
