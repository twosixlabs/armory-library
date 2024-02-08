import { computed } from 'vue';
import { useRoute } from "../stores/route.js";

const NavLink = {
    props: {
        route: String,
    },
    setup(props) {
        const routes = useRoute();
        const onClick = () => routes.setRoute(props.route);

        const classes = computed({
            get() {
                return {
                    'bg-twosix-green': props.route == routes.route,
                };
            },
        });

        return { classes, onClick };
    },
    template: `
        <a
            @click.prevent="onClick"
            :class="classes"
            class="hover:bg-twosix-grey hover:cursor-pointer my-auto p-2 rounded text-medium uppercase"
        >
            <slot></slot>
        </a>
    `,
};

export default {
    components: {
        NavLink,
    },
    template: `
        <div class="flex flex-row">
            <h1 class="font-medium text-xl mr-4 my-auto text-twosix-black uppercase">
                Evaluation Report
            </h1>
            <nav-link route="summary">Summary</nav-link>
            <nav-link route="parameters">Parameters</nav-link>
            <nav-link route="metrics">Metrics</nav-link>
        </div>
        <div>
            <img src="./assets/img/armory.png" alt="Armory" class="h-12" />
        </div>
    `
};
