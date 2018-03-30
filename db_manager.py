import pymysql


class DBManager(object):

    def __init__(self, config):
        self.conn = pymysql.connect(host=config.host,
                                    port=config.port,
                                    user=config.user,
                                    password=config.password,
                                    db=config.db,
                                    charset='utf8')
        self.curs = self.conn.cursor()


    def select_query(self, sql):
        self.curs.execute(sql)
        return self.curs
