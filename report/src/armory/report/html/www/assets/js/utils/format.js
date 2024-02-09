import dayjs from 'dayjs';

export const formatDuration = (duration) => dayjs.duration(duration).format('HH:mm:ss');

export const humanizeDuration = (duration) => dayjs.duration(duration).humanize();

export const formatTime = (time) => dayjs(time).toISOString();

export const humanizeTime = (time) => dayjs(time).fromNow();
