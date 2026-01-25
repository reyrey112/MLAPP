import psycopg2
from psycopg2 import sql
import logging
import pandas as pd


def create_database(db_name, user, password, host, port):
    conn = None

    try:
        conn = psycopg2.connect(
            database="postgres", user=user, password=password, host=host, port=port
        )

        conn.autocommit = True
        cursor = conn.cursor()

        create_db_query = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))

        cursor.execute(create_db_query)
        logging.warning(f"database {db_name} created ")

        cursor.close()

    except psycopg2.Error as e:
        print(f"erro: {e}")

    finally:
        if conn is not None:
            conn.close()


def create_tables(db_name, user, password, host, port):
    conn = None

    try:

        conn = psycopg2.connect(
            database=db_name,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        conn.autocommit = True

        cur = conn.cursor()

        commands = """
            CREATE TABLE formulation_table (
                distinguishing_ID INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                Formulation_Number SMALLINT,
                Main_Formulation_Number SMALLINT,
                Sub_Formulation_Number SMALLINT,
                Mixing_Time REAL,
                Active_1 REAL,
                Active_2 REAL,
                RM_3 REAL,
                RM_4 REAL,
                Active_3 REAL,
                RM_6 REAL,
                RM_7 REAL,
                RM_8 REAL,
                Active_4 REAL,
                Water REAL,
                Crashout REAL,
                Viscosity INTEGER,
                Batch_Number SMALLINT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """

        for command in commands:
            cur.execute(command)

        conn.commit()
        print("Tables created successfully")

        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error creating table: {error}")

    finally:
        if conn is not None:
            conn.close()


def connect_to_database(db_name, user, password, host, port):
    conn = None

    try:
        conn = psycopg2.connect(
            database=db_name,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        conn.autocommit = True

        cur = conn.cursor()

    except Exception as e:
        raise e

    return conn, cur


def table_exists_check(db_name, user, password, host, port):
    conn, cur = connect_to_database(db_name, user, password, host, port)

    query = """
        SELECT EXISTS (
            SELECT 1
            FROM)"""


def write_to_DB(db_name, user, password, host, port, table_name, data: pd.DataFrame):
    with psycopg2.connect(
        database=db_name, user=user, password=password, host="172.18.0.2", port = 5432
    ) as conn:
        with conn.cursor() as cur:

            data_tuple = [tuple(row) for row in data.itertuples(index=False)]

            cur.executemany(
                f"INSERT INTO {table_name} (mixing_time, active_1, active_2, rm_3, rm_4, active_3, rm_6, rm_7, rm_8, active_4, water, crashout, viscosity) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                data_tuple,
            )
            conn.commit()
    
