import pymysql

# 数据库连接配置
host = '114.132.83.76'
port = 3306
user = 'root'
password = '1234@Aliyun'
db_name = 'happiness_system'


def create_database():
    try:
        # 连接 MySQL (不指定数据库)
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset='utf8mb4'
        )
        cursor = conn.cursor()

        # 创建数据库
        print(f"Checking database '{db_name}'...")
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"Database '{db_name}' ensured.")

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Error creating database: {e}")


if __name__ == '__main__':
    create_database()