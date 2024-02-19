import { defineStore } from 'pinia';
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useEvaluationData } from './evaluation-data.js';

export const useMetricsSettings = defineStore('metrics-settings', () => {
    const evaluationData = useEvaluationData();

    const router = useRouter();
    const route = router.currentRoute;
    const updateQuery = (query) => {
        router.push({ query: { ...route.value.query, ...query } }, { replace: true });
    };

    // -- baseline chain

    const baselineChain = computed({
        get() {
            if (route.value.query.baselineChain == undefined) {
                return evaluationData.settings.baseline_chain;
            }
            return route.value.query.baselineChain;
        },
        set(baselineChain) {
            updateQuery({ baselineChain });
        },
    });

    function toggleBaselineChain(name) {
        if (baselineChain.value == name) {
            baselineChain.value = "";
        } else {
            baselineChain.value = name;
        }
    };

    // -- baseline run

    const baselineRun = computed({
        get() {
            if (route.value.query.baselineRun == undefined) {
                return evaluationData.settings.baseline_run;
            }
            return route.value.query.baselineRun;
        },
        set(baselineRun) {
            updateQuery({ baselineRun });
        },
    });

    function toggleBaselineRun(name) {
        if (baselineRun.value == name) {
            baselineRun.value = "";
        } else {
            baselineRun.value = name;
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
        const hideMetric = route.value.query.hideMetric;
        if (hideMetric == undefined) {
            return evaluationData.settings.hide_metrics || [];
        }
        if (Array.isArray(hideMetric)) {
            return hideMetric.filter((m) => m != "");
        }
        if (hideMetric) {
            return [hideMetric];
        }
        return [];
    });

    function toggleMetric(metric) {
        let hideMetric = [...hiddenMetrics.value]; // make a copy
        if (hideMetric.includes(metric)) {
            const index = hideMetric.indexOf(metric);
            hideMetric.splice(index, 1);
            if (hideMetric.length == 0) {
                hideMetric.push("");
            }
        } else {
            hideMetric.push(metric);
            hideMetric = hideMetric.filter((h) => h != "");
        }

        updateQuery({ hideMetric });
    }

    // -- chain visibility

    const hiddenChains = computed(() => {
        const hideChain = route.value.query.hideChain;
        if (hideChain == undefined) {
            return evaluationData.settings.hide_chains || [];
        }
        if (Array.isArray(hideChain)) {
            return hideChain.filter((c) => c != "");
        }
        if (hideChain) {
            return [hideChain];
        }
        return [];
    });

    function toggleChain(chain) {
        let hideChain = [...hiddenChains.value]; // make a copy
        if (hideChain.includes(chain)) {
            const index = hideChain.indexOf(chain);
            hideChain.splice(index, 1);
            if (hideChain.length == 0) {
                hideChain.push("");
            }
        } else {
            hideChain.push(chain);
            hideChain = hideChain.filter((h) => h != "");
        }

        updateQuery({ hideChain });
    }

    // -- parameter visibility

    const showParameters = computed(() => {
        const showParameter = route.value.query.showParameter;
        if (showParameter == undefined) {
            return evaluationData.settings.show_parameters || [];
        }
        if (Array.isArray(showParameter)) {
            return showParameter.filter((p) => p != "");
        }
        if (showParameter) {
            return [showParameter];
        }
        return [];
    });

    function toggleParameter(parameter) {
        let showParameter = [...showParameters.value]; // make a copy
        if (showParameter.includes(parameter)) {
            const index = showParameter.indexOf(parameter);
            showParameter.splice(index, 1);
            if (showParameter.length == 0) {
                showParameter.push("");
            }
        } else {
            showParameter.push(parameter);
            showParameter = showParameter.filter((h) => h != "");
        }

        updateQuery({ showParameter });
    }

    // -- show-all visibility

    const showAll = computed({
        get() {
            const showAll = route.value.query.showAll;
            if (showAll == undefined) {
                return true;
            }
            return showAll == "true";
        },
        set(showAll) {
            updateQuery({ showAll });
        },
    });

    return {
        baselineChain,
        baselineRun,
        getMetricType,
        hiddenChains,
        hiddenMetrics,
        precision,
        setMetricType,
        showAll,
        showParameters,
        toggleBaselineChain,
        toggleBaselineRun,
        toggleChain,
        toggleMetric,
        toggleParameter,
    };
});
