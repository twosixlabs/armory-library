import Overview from './overview.js';
import SingleRun from './single-run.js';
import MultipleRuns from './multiple-runs.js';

export const routes = [
    { path: '/', component: Overview },
    { path: '/run/:id', component: SingleRun, props: true },
    { path: '/compare', component: MultipleRuns },
];
