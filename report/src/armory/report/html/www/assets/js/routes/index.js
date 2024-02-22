import CompareMetrics from './compare-metrics.js';
import CompareRuntimes from './compare-runtimes.js';
import MultipleRuns from './multiple-runs.js';
import Overview from './overview.js';
import RunDetails from './run-details.js';
import RunDiff from './run-diff.js';
import RunFlowchart from './run-flowchart.js';
import RunMetrics from './run-metrics.js';
import RunParameters from './run-parameters.js';
import RunRuntime from './run-runtime.js';
import RunSamples from './run-samples.js';
import SingleRun from './single-run.js';

export const routes = [
    { path: '/', name: 'index', component: Overview },
    {
        path: '/run/:id',
        name: 'single-run',
        component: SingleRun,
        redirect: { name: 'single-run-details' },
        props: true,
        children: [
            { path: '', name: 'single-run-details', component: RunDetails },
            { path: 'flowchart', name: 'single-run-flowchart', component: RunFlowchart },
            { path: 'metrics', name: 'single-run-metrics', component: RunMetrics },
            { path: 'parameters', name: 'single-run-params', component: RunParameters },
            { path: 'runtime', name: 'single-run-runtime', component: RunRuntime },
            { path: 'samples', name: 'single-run-samples', component: RunSamples },
        ],
    },
    {
        path: '/compare',
        name: 'compare-runs',
        component: MultipleRuns,
        children: [
            { path: '', redirect: { name: 'compare-runs-metrics' } },
            { path: 'diff', name: 'compare-runs-diff', component: RunDiff },
            { path: 'metrics', name: 'compare-runs-metrics', component: CompareMetrics },
            { path: 'runtime', name: 'compare-runs-runtime', component: CompareRuntimes },
        ],
    },
];
