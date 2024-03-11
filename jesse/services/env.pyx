from dotenv import load_dotenv, dotenv_values
#import jesse.helpers as jh
import os
import sys

ls = os.listdir('.')
# fix directory issue
sys.path.insert(0, os.getcwd())

ENV_VALUES = {}

if 'strategies' in ls and 'storage' in ls:
    # load env
    load_dotenv()

    # create and expose ENV_VALUES
    ENV_VALUES = dotenv_values('.env')

    if "pytest" in sys.modules:
        ENV_VALUES['POSTGRES_HOST'] = '127.0.0.1'
        ENV_VALUES['POSTGRES_NAME'] = 'jesse_db'
        ENV_VALUES['POSTGRES_PORT'] = '5432'
        ENV_VALUES['POSTGRES_USERNAME'] = 'jesse_user'
        ENV_VALUES['POSTGRES_PASSWORD'] = 'password'
        ENV_VALUES['REDIS_HOST'] = 'localhost'
        ENV_VALUES['REDIS_PORT'] = '6379'
        ENV_VALUES['REDIS_DB'] = 0
        ENV_VALUES['REDIS_PASSWORD'] = ''
        ENV_VALUES['APP_PORT'] = '9000'

    # validation for existence of .env file
    if len(list(ENV_VALUES.keys())) == 0:
        import jesse.helpers as jh
        jh.error(
            '.env file is missing from within your local project. '
            'This usually happens when you\'re in the wrong directory. '
            '\n\nIf you haven\'t created a Jesse project yet, do that by running: \n'
            'jesse make-project {name}\n'
            'And then go into that project, and run the same command.',
            force_print=True
        )
        os._exit(1)
        jh.terminate_app()
        # raise FileNotFoundError('.env file is missing from within your local project. This usually happens when you\'re in the wrong directory. You can create one by running "cp .env.example .env"')

    if not "pytest" in sys.modules and ENV_VALUES['PASSWORD'] == '':
        raise EnvironmentError('You forgot to set the PASSWORD in your .env file')
