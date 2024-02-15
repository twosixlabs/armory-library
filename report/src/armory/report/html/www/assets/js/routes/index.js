import CompareMetrics from './compare-metrics.js';
import MultipleRuns from './multiple-runs.js';
import Overview from './overview.js';
import RunArtifacts from './run-artifacts.js';
import RunDetails from './run-details.js';
import RunDiff from './run-diff.js';
import RunMetrics from './run-metrics.js';
import RunParameters from './run-parameters.js';
import RunPipeline from './run-pipeline.js';
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
            { path: 'metrics', name: 'single-run-metrics', component: RunMetrics },
            { path: 'parameters', name: 'single-run-params', component: RunParameters },
            { path: 'pipeline', name: 'single-run-pipeline', component: RunPipeline },
            { path: 'artifacts', name: 'single-run-artifacts', component: RunArtifacts },
        ],
    },
    {
        path: '/compare',
        name: 'compare-runs',
        component: MultipleRuns,
        children: [
            { path: '', redirect: { name: 'compare-runs-metrics' } },
            { path: 'metrics', name: 'compare-runs-metrics', component: CompareMetrics },
            { path: 'diff', name: 'compare-runs-diff', component: RunDiff },
        ],
    },
];
