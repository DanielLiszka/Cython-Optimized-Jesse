#!/bin/bash

# This script will create a postgresql database on ramdisk
# create jesse_db, jesse user, and grant privileges

# Stop postgresql service if it is running
echo "Trying to stop Postgresql service gracefully, watch ps aux output for it..."
sudo systemctl stop postgresql
sudo service postgresql stop  # WSL2
sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /tmp/ramdisk/db/ stop
sudo umount /tmp/ramdisk/
sleep 10

ps aux | grep postgresql
read -rsp $'Press any key to continue...\n' -n1 key

# Create ramdisk, mount it, set permissions
sudo mkdir /tmp/ramdisk
sudo chmod 777 /tmp/ramdisk
sudo mount -t tmpfs -o size=8G ramdisk /tmp/ramdisk
mount | tail -n 1
mkdir /tmp/ramdisk/db
sudo chmod 0700 -R /tmp/ramdisk/db
sudo chown -R postgres:postgres /tmp/ramdisk/db

# Initialize db directory as Postgresql database
sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /tmp/ramdisk/db initdb

# Start Postgresql with custom directory
sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /tmp/ramdisk/db -w start

# Create jesse_db, user and grant privileges
sudo -u postgres psql -c "CREATE DATABASE jesse_db;"
sudo -u postgres psql -c "CREATE USER jesse_user WITH PASSWORD 'password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE jesse_db to jesse_user;"

# Warmup jesse_db
jesse import-candles Binance btc-usdt 2021-12-04 --skip-confirmation

# Batch import
jesse-tk bulk spot btc-usdt 2018-02-01 --workers 8
jesse-tk bulk spot eth-usdt 2018-02-01 --workers 8
jesse-tk bulk spot bnb-usdt 2018-02-01 --workers 8

# 2nd pass with jesse import-candles to fill missing candles
jesse import-candles Binance btc-usdt 2018-02-09 --skip-confirmation
jesse import-candles Binance eth-usdt 2018-02-09 --skip-confirmation
jesse import-candles Binance bnb-usdt 2018-02-09 --skip-confirmation

# Vacuum, analyze and reindex jesse_db
sudo -u postgres /usr/lib/postgresql/14/bin/vacuumdb  --analyze --verbose -d jesse_db -e -f
sudo -u postgres /usr/lib/postgresql/14/bin/reindexdb --verbose jesse_db