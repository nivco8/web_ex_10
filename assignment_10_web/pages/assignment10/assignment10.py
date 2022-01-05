from flask import Blueprint, render_template, request, redirect, flash
from interact_with_DB import interact_db

assignment10 = Blueprint('assignment10', __name__,
                         static_folder='static',
                         static_url_path='/assignment10',
                         template_folder='templates')


@assignment10.route('/assignment10', methods=['GET', 'POST'])
def users_func():
    query = 'select * from users;'
    users = interact_db(query=query, query_type='fetch')
    return render_template('/assignment10.html', users=users)


@assignment10.route('/new_user', methods=['POST'])
def add_func():
    first_name = request.form["first_name"]
    last_name = request.form["last_name"]
    email = request.form["email"]
    query = "INSERT INTO users(first_name, last_name, email) VALUES ('%s', '%s', '%s');" % (
        first_name, last_name, email)
    interact_db(query=query, query_type='commit')
    flash(f'User {first_name} {last_name} inserted successfully.')
    return redirect('/assignment10')


@assignment10.route('/update_user', methods=['POST'])
def update_func():
    user_id = request.form['id']
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    email = request.form['email']
    query = "select * FROM users WHERE id = '%s';" % user_id
    query_result = interact_db(query=query, query_type='fetch')
    if len(query_result) > 0:
        query = "UPDATE users SET first_name='%s', last_name='%s', email='%s' WHERE id = '%s';" % (first_name, last_name, email, user_id)
        interact_db(query=query, query_type='commit')
        flash('user updated successfully.')
        return redirect('/assignment10')
    else:
        flash(f'User {user_id} does not exist. No changes.')
        return redirect('/assignment10')


@assignment10.route('/remove_user', methods=['POST'])
def remove_func():
    user_id = request.form['id']
    query = "DELETE FROM users WHere id='%s'" % user_id
    interact_db(query=query, query_type='commit')
    flash(f'User {user_id} deleted successfully.')
    return redirect('/assignment10')

