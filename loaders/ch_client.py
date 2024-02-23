from clickhouse_driver import Client
import pandas as pd

"""
HOST IP:
- local (local laptop instance): '3.138.222.138'
- aws (sagemaker, lambdas, etc..): '172.31.9.89'
"""

# Get ClickHouse client
# ---------------------
def get_client(
    host_ip: str = '3.138.222.138',
    port: str = '9000', 
    database: str = 'test', 
    user: str = 'default'
) -> object:

    client = Client(
        host=host_ip,
        port=port,
        database=database,
        user=user,
        compression=True,
        connect_timeout=500_000,
        send_receive_timeout=500_000,
        sync_request_timeout=500_000
    )

    return client

# Execute SQL query
# -----------------
def execute_query(
    run_locally: bool,
    query: str, 
    col_names: list[str]
) -> pd.DataFrame:

    client = get_client(run_locally=run_locally)

    try:
        result = client.execute(query)
        df = pd.DataFrame(result, columns=col_names)

        return df

    except Exception as e:
        print('Something went wrong when executing the query...', e)
        print(query)

        return None