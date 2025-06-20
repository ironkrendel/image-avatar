import multiprocessing

class Logger():
    def __init__(self, data_folder, data_count):
        self.data_folder = data_folder
        self.data_count = data_count
        self.counter = 0
        self.queue = []
        self.successful_frames = []

        with open('tmp_metadata_cache.txt', 'w') as f:
            f.write("")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def process_queue(self):
        if len(self.queue) <= 0:
            return

        event = self.queue.pop(0)

        if event['operation'] == 'success':
            print('\033[92m' + f"{1000 * event['time']:.2f}ms {self.counter / self.data_count * 100:.2f}% {event['filename']}" + '\033[0m')

            json_filename = event['filename'].split('.')
            json_filename[-1] = 'json'
            json_filename = '.'.join(json_filename)
            log_file = open('tmp_metadata_cache.txt', 'a')
            log_file.write(f'{json_filename}\n')
            log_file.close()

            self.counter += 1
        elif event['operation'] == 'failure':
            print('\033[91m' + f"{1000 * event['time']:.2f}ms {self.counter / self.data_count * 100:.2f}% {event['filename']}" + '\033[0m')

            self.counter += 1
        elif event['operation'] == 'error':
            print('\033[1;31m' + " Error on file " + event['filename'] + '\033[0m')

            self.counter += 1

        self.process_queue()

    def log_success(self, time, filename):
        # print('\033[92m' + f"{1000 * time:.2f}ms {self.counter / self.data_count * 100:.2f}% {filename}" + '\033[0m')
        self.queue.append({
            'operation': 'success',
            'time': time,
            'filename': filename,
        })
        self.process_queue()

    def log_failure(self, time, filename):
        # print('\033[91m' + f"{1000 * time:.2f}ms {self.counter / self.data_count * 100:.2f}% {filename}" + '\033[0m')
        self.queue.append({
            'operation': 'failure',
            'time': time,
            'filename': filename,
        })
        self.process_queue()

    def log_error(self, filename, error):
        self.queue.append({
            'operation': 'error',
            'filename': filename,
            'error': error,
        })
        self.process_queue()