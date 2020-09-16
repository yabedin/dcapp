class Config(object):
    DEBUG = False
    TESTING = False

    DB_NAME = "production-db"
    DB_USERNAME = "user"
    DB_PASSWORD = "exampledb"

    ALLOWED_FILE_EXTENSIONS = ["JSON"]

    SESSION_COOKIE_SECURE = True

    SECRET_KEY = "randomsecretkey"

class ProductionConfig(Config):
    pass 

class DevelopmentConfig(Config):
    DEBUG = True

    SESSION_COOKIE_SECURE = False
    DB_NAME = "development-db"

    JSON_LOCATION = "/Users/yasminabedin/Documents/msc-cs/dcapp/app/static/uploads/json"
    CSV_LOCATION = "/Users/yasminabedin/Documents/msc-cs/dcapp/app/static/uploads/csv"

class TestingConfig(Config):
    TESTING = True

    SESSION_COOKIE_SECURE = False

