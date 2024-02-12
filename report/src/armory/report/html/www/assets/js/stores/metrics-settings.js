import { defineStore } from 'pinia';
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useEvaluationData } from './evaluation-data.js';

export const useMetricsSettings = defineStore('metrics-settings', () => {
    const evaluationData = useEvaluationData();

    const router = useRouter();
    const route = router.currentRoute;
    const updateQuery = (query) => {
        router.push({
            path: route.value.path,
            query: { ...route.value.query, ...query },
        }, { replace: true }).catch((err) => console.log(err));
    };

    // -- baseline chain

    const baseline = computed({
        get() {
            if (route.value.query.baseline == undefined) {
                return evaluationData.settings.baseline_chain;
            }
            return route.value.query.baseline;
        },
        set(baseline) {
            updateQuery({ baseline });
        },
    });

    function toggleBaseline(name) {
        if (baseline.value == name) {
            baseline.value = "";
        } else {
            baseline.value = name;
        }
    };

    // -- metric precision

    const precision = computed({
        get() {
            const p = route.value.query.precision || evaluationData.settings.metric_precision;
            return p ? Number.parseInt(p) : 3;
        },
        set(precision) {
            updateQuery({ precision });
        },
    });

    // -- metric comparison type

    function getMetricType(metric) {
        return route.value.query[`metric.${metric}`] || evaluationData.settings.metric_types[metric] || "high";
    }

    function setMetricType(metric, metricType) {
        updateQuery({ [`metric.${metric}`]: metricType });
    }

    // -- metric visibility

    const hiddenMetrics = computed(() => {
        const hide = route.value.query.hide;
        if (hide == undefined) {
            return evaluationData.settings.hide_metrics || [];
        }
        if (Array.isArray(hide)) {
            return hide;
        }
        if (hide) {
            return [hide];
        }
        return [];
    });

    function toggleMetric(metric) {
        let hide = [...hiddenMetrics.value]; // make a copy
        if (hide.includes(metric)) {
            const index = hide.indexOf(metric);
            hide.splice(index, 1);
            if (hide.length == 0) {
                hide.push("");
            }
        } else {
            hide.push(metric);
            hide = hide.filter((h) => h != "");
        }

        updateQuery({ hide });
    }

    return {
        baseline,
        getMetricType,
        hiddenMetrics,
        precision,
        setMetricType,
        toggleBaseline,
        toggleMetric,
    };
});
