import startup
from watcher.folder_watcher import FolderWatcher


if __name__ == "__main__":

    watcher = FolderWatcher()

    watcher.run()