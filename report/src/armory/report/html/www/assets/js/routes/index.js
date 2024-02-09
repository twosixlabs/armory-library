import Overview from './overview.js';
import SingleRun from './single-run.js';
import MultipleRuns from './multiple-runs.js';

export const routes = [
    { path: '/', component: Overview },
    { path: '/run/:id', component: SingleRun },
    { path: '/compare', component: MultipleRuns },
];
