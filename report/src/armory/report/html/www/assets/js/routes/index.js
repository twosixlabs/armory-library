import Overview from './overview.js';
import SingleRun from './single-run.js';
import MultipleRuns from './multiple-runs.js';

export const routes = [
    { path: '/', name: 'index', component: Overview },
    {
        path: '/run/:id',
        name: 'single-run',
        component: SingleRun,
        props: true,
    },
    { path: '/compare', name: 'compare-runs', component: MultipleRuns },
];
