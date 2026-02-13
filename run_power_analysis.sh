#!/bin/bash
# run_power_short.sh — оптимизированный запуск power analysis на Threadripper

cd /projects/TFNBS  # Рабочая директория

# Оптимальные env для 16 ядер (из вашего теста)
#export OMP_NUM_THREADS=16
#export MKL_NUM_THREADS=16
#export OPENBLAS_NUM_THREADS=16

# Лог и PID файлы
LOG=/projects/TFNBS/.log/power_analysis_short_$(date +%d%m%H%M).log
PIDFILE=/projects/TFNBS/.log/power_analysis_short_$(date +%d%m%H%M).pid

# Удаляем старые (если есть)
rm -f $PIDFILE $LOG

# Запуск в фоне
nohup taskset -c 0-31 \
  python -u examples/power_test/power_analysis.py \
  --config examples/power_test/sweep_config_power_server_short.yaml --resume \
  > $LOG 2>&1 &

# PID в файл
echo $! > $PIDFILE

echo "Запущен PID $(cat $PIDFILE)"
echo "Лог: $LOG"
echo "Мониторинг: tail -f $LOG | grep -E 'run|TPR|ETA|ERROR'"
echo "Остановить: kill $(cat $PIDFILE)"
echo "Статус: htop -p $(cat $PIDFILE)"
