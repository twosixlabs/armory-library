import { defineStore } from 'pinia';
import { computed } from 'vue';
import { useRouter } from 'vue-router';

export const useMetricsSettings = defineStore('metrics-settings', () => {
    const router = useRouter();
    const route = router.currentRoute;
    const updateQuery = (query) => {
        router.push({
            path: route.value.path,
            query: { ...route.value.query, ...query },
        }, { replace: true }).catch((err) => console.log(err));
    };

    // -- baseline metric

    const baseline = computed({
        get() {
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
            const p = route.value.query.precision;
            return p ? Number.parseInt(p) : 3;
        },
        set(precision) {
            updateQuery({ precision });
        },
    });

    // -- metric comparison type

    function getMetricType(metric) {
        return route.value.query[`metric.${metric}`] || "high";
    }

    function setMetricType(metric, metricType) {
        updateQuery({ [`metric.${metric}`]: metricType });
    }

    // -- metric visibility

    const hiddenMetrics = computed(() => {
        const hide = route.value.query.hide;
        if (Array.isArray(hide)) {
            return hide;
        }
        if (hide) {
            return [hide];
        }
        return [];
    });

    function toggleMetric(metric) {
        let hide = route.value.query.hide;
        console.log({ metric, hide })
        if (Array.isArray(hide)) {
            hide = [...hide]; // make a copy
            if (hide.includes(metric)) {
                const index = hide.indexOf(metric);
                console.log('removing from array', index)
                hide.splice(index, 1);
                if (hide.length == 0) {
                    hide = undefined;
                }
            } else {
                console.log('adding to array')
                hide.push(metric);
            }
        } else if (hide == metric) { // remove value as single entry
            hide = undefined;
        } else if (hide) { // add value to existing single entry
            hide = [hide, metric];
        } else { // set value as single entry
            hide = [metric];
        }

        console.log('new', { hide })
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
