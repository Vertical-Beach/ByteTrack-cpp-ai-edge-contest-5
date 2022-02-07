
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <time.h>
#include <stdlib.h>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <cstring>

int reset_pl_resetn0(){
	int fd;
	char attr[32];

	DIR *dir = opendir("/sys/class/gpio/gpio510");
	if (!dir) {
		fd = open("/sys/class/gpio/export", O_WRONLY);
		if (fd < 0) {
			perror("open(/sys/class/gpio/export)");
			return -1;
		}
		strcpy(attr, "510");
		write(fd, attr, strlen(attr));
		close(fd);
		dir = opendir("/sys/class/gpio/gpio510");
		if (!dir) {
			return -1;
		}
	}
	closedir(dir);

	fd = open("/sys/class/gpio/gpio510/direction", O_WRONLY);
	if (fd < 0) {
		perror("open(/sys/class/gpio/gpio510/direction)");
		return -1;
	}
	strcpy(attr, "out");
	write(fd, attr, strlen(attr));
	close(fd);

	fd = open("/sys/class/gpio/gpio510/value", O_WRONLY);
	if (fd < 0) {
		perror("open(/sys/class/gpio/gpio510/value)");
		return -1;
	}
	sprintf(attr, "%d", 0);
	write(fd, attr, strlen(attr));

    sprintf(attr, "%d", 1);
	write(fd, attr, strlen(attr));
	close(fd);

	return 0;
}