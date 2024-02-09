import Overview from './overview.js';
import SingleRun from './single-run.js';

export const routes = [
    { path: '/', component: Overview },
    { path: '/run/:id', component: SingleRun },
];
